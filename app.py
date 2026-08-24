import os
os.environ['NO_PROXY'] = '*'

import streamlit as st

# 🚨 CRITICAL FIX: set_page_config MUST be the very first Streamlit command executed!
st.set_page_config(page_title="Textbook Index Extractor", layout="wide")

import pdfplumber
import pandas as pd
import json
import io
import zipfile
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# --- 1. CONFIGURATION & AUTHENTICATION ---
load_dotenv()

# Safely pull keys from Streamlit Secrets (Cloud) or fallback to local .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if st.secrets.get("OPENAI_API_KEY"):
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
except Exception:
    # Safely ignored for local runs where secrets.toml doesn't exist
    pass

# Initialize OpenAI Client (Bypassing Windows/Cloud Proxies)
if OPENAI_API_KEY:
    try:
        custom_http = httpx.Client(proxy=None, timeout=60.0)
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY, 
            http_client=custom_http,
            max_retries=3
        )
    except Exception as e:
        openai_client = None
        st.error(f"OpenAI Initialization Error: {e}")
else:
    openai_client = None

# --- 2. TEXT & PDF EXTRACTION FUNCTIONS ---
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def extract_pdf_streams(uploaded_files):
    pdf_streams = []
    for file in uploaded_files:
        filename = file.name.lower()
        if filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(file) as zf:
                    valid_pdf_names = [
                        name for name in zf.namelist() 
                        if name.lower().endswith(".pdf") 
                        and not name.startswith("__MACOSX/") 
                        and not name.split("/")[-1].startswith("._")
                    ]
                    valid_pdf_names.sort(key=natural_sort_key)
                    for name in valid_pdf_names:
                        pdf_bytes = io.BytesIO(zf.read(name))
                        pdf_bytes.name = name.split("/")[-1]
                        pdf_streams.append(pdf_bytes)
            except Exception as e:
                st.error(f"Error reading ZIP file {file.name}: {e}")
        elif filename.endswith(".pdf"):
            pdf_streams.append(file)
            
    pdf_streams.sort(key=lambda x: natural_sort_key(x.name))
    return pdf_streams

def extract_text_from_pdf(pdf_file_obj):
    full_text = []
    with pdfplumber.open(pdf_file_obj) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"--- PAGE {i+1} ---\n" + text)
    return "\n\n".join(full_text)

def process_text_with_ai(raw_text, filename, max_retries=3):
    """Passes chapter text AND filename to GPT-4o-mini to extract structured data."""
    if not openai_client:
        st.error(f"🚨 OpenAI Client is not initialized. Please check your API key.")
        return []
        
    prompt = f"""
    You are an expert curriculum and textbook indexing system. I am providing you with the text of a school textbook chapter.
    
    Source File Name: {filename}
    
    Your task is to comprehensively extract the Chapter Number, Chapter Name, and ALL OF ITS SUBTOPICS.
    
    CRITICAL INSTRUCTIONS:
    1. Extract the exact Chapter Number and Chapter Title. 
       - **CRUCIAL RULE:** Use the 'Source File Name' provided above as the ultimate source of truth for the Chapter Number (e.g., if the file is named 'ch11_something.pdf', the Chapter Number MUST be "11"). Do not let stray page numbers or typos in the text confuse you.
    2. Extract all main subtopics and section headings.
    3. Generate sequential Subtopic IDs starting with the exact Chapter Number (e.g., 11.1, 11.2, 11.3).
    4. Ignore questions, exercises, 'Let's Check', 'Activities', 'Did You Know' sidebars, and generic page filler.
    
    Output STRICTLY a valid JSON array of objects. Do not include markdown formatting like ```json.
    Format each item exactly like this:
    [
      {{
        "Chapter Number": "1",
        "Chapter Name": "Locating Places and Reading Maps",
        "Subtopic ID": "1.1",
        "Subtopic Name": "Shape of Earth"
      }}
    ]
    
    Textbook Content:
    {raw_text}
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Always output clean, raw JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content.strip())
            
        except Exception as e:
            if attempt < max_retries: 
                time.sleep(3 * attempt)
            else: 
                st.error(f"🚨 Text Extraction API Error on {filename}: {repr(e)}")
                return []

# --- 3. STREAMLIT UI & MAIN PIPELINE ---
st.title("📚 Textbook Index Extractor")
st.markdown("Automated curriculum text parser mapped directly to structured Excel sheets.")

subject_input = st.text_input("Subject Name", value="", placeholder="e.g. Maths, Science, Social Studies...")

uploaded_files = st.file_uploader("Upload Textbook PDFs or ZIP files", type=["pdf", "zip"], accept_multiple_files=True)

if uploaded_files and st.button("Extract Data & Generate Master Excel", type="primary"):
    discovered_pdfs = extract_pdf_streams(uploaded_files)
    master_data = []
    
    progress_bar = st.progress(0, text="Starting text extraction...")
    
    for idx, pdf_file in enumerate(discovered_pdfs):
        progress_bar.progress((idx + 1) / len(discovered_pdfs), text=f"Processing `{pdf_file.name}`...")
        
        # Step 1: Text Extraction
        raw_text = extract_text_from_pdf(pdf_file)
        
        if len(raw_text.strip()) < 50:
            st.warning(f"⚠️ `{pdf_file.name}` has no selectable text (it might be a scanned image). Skipping.")
            continue
            
        chapter_data = process_text_with_ai(raw_text, pdf_file.name)
        
        if not chapter_data:
            st.warning(f"⚠️ No structural data could be extracted from `{pdf_file.name}`. Skipping.")
            continue
            
        # Step 2: Format rows into the required columns (SUBJECT, MODULE, CHAPTER)
        for item in chapter_data:
            formatted_row = {
                "SUBJECT": subject_input,
                "MODULE": item.get("Chapter Name", ""),
                "CHAPTER": item.get("Subtopic Name", "")
            }
            master_data.append(formatted_row)
            
        time.sleep(1.0)
        
    progress_bar.empty()
    
    if not master_data:
        st.error("❌ Critical Failure: Could not extract any data to build the Master Excel file.")
    else:
        df = pd.DataFrame(master_data)
        excel_buffer = io.BytesIO()
        
        # Write to Excel with clean formatting
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Master Index')
            worksheet = writer.sheets['Master Index']
            from openpyxl.utils import get_column_letter
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 3
                worksheet.column_dimensions[get_column_letter(i + 1)].width = max_len

        st.success(f"🎉 Complete! Processed {len(df['MODULE'].unique())} chapters successfully.")
        st.dataframe(df)
        
        st.download_button(
            label="📥 Download Master_Index.xlsx",
            data=excel_buffer.getvalue(),
            file_name="Master_Index.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )