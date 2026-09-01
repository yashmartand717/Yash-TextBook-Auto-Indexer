import os
os.environ['NO_PROXY'] = '*'

import streamlit as st

# 🚨 CRITICAL FIX: set_page_config MUST be the very first Streamlit command executed!
st.set_page_config(page_title="Textbook Index Extractor", layout="wide")

import pymupdf  # Replaced pdfplumber for extreme memory efficiency
import pandas as pd
import json
import io
import zipfile
import re
import time
import gc       # Used to manually flush server RAM
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# --- 1. CONFIGURATION & AUTHENTICATION ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if st.secrets.get("OPENAI_API_KEY"):
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
except Exception:
    pass

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


# --- 2. TEXT EXTRACTION & AI FUNCTIONS ---
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
    pdf_bytes = pdf_file_obj.read()
    
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text:
                full_text.append(f"--- PAGE {i+1} ---\n" + text)
                
    del pdf_bytes
    gc.collect()
    return "\n\n".join(full_text)


def process_text_with_ai(raw_text, filename, max_retries=3):
    """Extracts the granular structural data from the raw text."""
    if not openai_client:
        return []
        
    prompt = f"""
    You are an expert curriculum and book indexing system. I am providing you with the text of a book or textbook.
    
    Source File Name: {filename}
    
    Your task is to extract the structural hierarchy of this text into a JSON array. 
    
    CRITICAL INSTRUCTIONS:
    1. Identify the primary structural grouping of the book (e.g., "Unit", "Part", "Section", or "Chapter"). Extract this as "Primary_Grouping".
    2. Identify the secondary level within that grouping (e.g., "Chapters" inside a "Part", or "Subtopics" inside a "Chapter"). Extract this as "Secondary_Item".
    3. DO NOT invent numbering systems if they do not exist. Use exactly what is written.
    4. If the book is flat (just Chapters with no subtopics), put the Chapter Name in "Primary_Grouping" and leave "Secondary_Item" blank.
    5. Ignore generic filler like forewords or introductions.
    
    Output STRICTLY a valid JSON array of objects. Do not include markdown formatting like ```json.
    Format each item exactly like this:
    [
      {{
        "Primary_Grouping": "PART I - What should I eat?",
        "Secondary_Item": "Chapter 1 - Eat food."
      }}
    ]
    
    Book Content:
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
                temperature=0.1,
                max_tokens=8192 
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


def summarize_index_with_ai(detailed_data, max_retries=3):
    """Takes the granular index and groups micro-chapters into broader thematic modules."""
    if not openai_client:
        return []
        
    prompt = f"""
    You are an expert curriculum designer. I am providing you with a highly detailed, granular book index (JSON) containing many small chapters or rules.
    
    Your task is to group these granular items into a condensed, summarized index.
    
    CRITICAL INSTRUCTIONS:
    1. Group every 4 to 6 related granular items together into a broader, thematic "Summarized_Chapter".
    2. Assign these grouped chapters to a "Module" (You can use the existing 'MODULE' names or create broader thematic ones based on the content).
    3. Output STRICTLY a valid JSON array of objects. Do not include markdown formatting like ```json.
    
    Format exactly like this:
    [
      {{
        "MODULE": "Part I - What should I eat?",
        "CHAPTER": "Rules 1-6: Defining Real Food vs. Processed Products"
      }},
      {{
        "MODULE": "Part I - What should I eat?",
        "CHAPTER": "Rules 7-12: Navigating Supermarket Traps and Labels"
      }}
    ]
    
    Granular Index Data:
    {json.dumps(detailed_data, indent=2)}
    """

    for attempt in range(1, max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise data summarization assistant. Always output clean, raw JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096
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
                st.error(f"🚨 Summarization API Error: {repr(e)}")
                return []


# --- 3. STREAMLIT UI & MAIN PIPELINE ---
st.title("📚 Textbook Index Extractor")
st.markdown("Automated curriculum text parser mapped directly to structured Excel sheets.")

subject_input = st.text_input("Subject Name", value="", placeholder="e.g. Maths, Science, Social Studies...")
uploaded_files = st.file_uploader("Upload Textbook PDFs or ZIP files", type=["pdf", "zip"], accept_multiple_files=True)

# Checkbox for the new feature
generate_summary = st.checkbox("Also generate a Summarized Index (Groups 5-6 micro-chapters into broader themes)", value=True)

if uploaded_files and st.button("Extract Data & Generate Master Excel", type="primary"):
    discovered_pdfs = extract_pdf_streams(uploaded_files)
    master_data = []
    
    progress_bar = st.progress(0, text="Starting text extraction...")
    
    # 1. GRANULAR EXTRACTION LOOP
    for idx, pdf_file in enumerate(discovered_pdfs):
        progress_bar.progress((idx + 1) / len(discovered_pdfs), text=f"Processing `{pdf_file.name}`...")
        
        raw_text = extract_text_from_pdf(pdf_file)
        
        if len(raw_text.strip()) < 50:
            st.warning(f"⚠️ `{pdf_file.name}` has no selectable text. Skipping.")
            continue
            
        chapter_data = process_text_with_ai(raw_text, pdf_file.name)
        
        if not chapter_data:
            st.warning(f"⚠️ No structural data could be extracted from `{pdf_file.name}`. Skipping.")
            continue
            
        for item in chapter_data:
            formatted_row = {
                "SUBJECT": subject_input,
                "MODULE": item.get("Primary_Grouping", ""),
                "CHAPTER": item.get("Secondary_Item", "")
            }
            master_data.append(formatted_row)
            
        time.sleep(1.0)
    
    # 2. OPTIONAL SUMMARIZATION PASS
    summary_data = []
    if generate_summary and master_data:
        progress_bar.progress(1.0, text="Grouping micro-chapters into summaries...")
        raw_summary = summarize_index_with_ai(master_data)
        
        for item in raw_summary:
            summary_row = {
                "SUBJECT": subject_input,
                "MODULE": item.get("MODULE", ""),
                "CHAPTER": item.get("CHAPTER", "")
            }
            summary_data.append(summary_row)
            
    progress_bar.empty()
    
    # 3. RENDER UI & EXCEL GENERATION
    if not master_data:
        st.error("❌ Critical Failure: Could not extract any data.")
    else:
        df_detailed = pd.DataFrame(master_data)
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Write Sheet 1
            df_detailed.to_excel(writer, index=False, sheet_name='Detailed Index')
            worksheet_det = writer.sheets['Detailed Index']
            from openpyxl.utils import get_column_letter
            for i, col in enumerate(df_detailed.columns):
                max_len = max(df_detailed[col].astype(str).map(len).max(), len(str(col))) + 3
                worksheet_det.column_dimensions[get_column_letter(i + 1)].width = max_len
                
            # Write Sheet 2 if selected
            if generate_summary and summary_data:
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, index=False, sheet_name='Summarized Index')
                worksheet_sum = writer.sheets['Summarized Index']
                for i, col in enumerate(df_summary.columns):
                    max_len = max(df_summary[col].astype(str).map(len).max(), len(str(col))) + 3
                    worksheet_sum.column_dimensions[get_column_letter(i + 1)].width = max_len

        st.success(f"🎉 Complete! Processed {len(df_detailed)} total entries.")
        
        # Display Tabs for easy viewing
        if generate_summary and summary_data:
            tab1, tab2 = st.tabs(["Detailed Index", "Summarized Index"])
            with tab1:
                st.dataframe(df_detailed)
            with tab2:
                st.dataframe(df_summary)
        else:
            st.dataframe(df_detailed)
        
        st.download_button(
            label="📥 Download Master_Index.xlsx",
            data=excel_buffer.getvalue(),
            file_name="Master_Index.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )