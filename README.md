# Document Data Extractor (Textbook Auto-Indexer)

A Streamlit app that parses curriculum documents and lab manuals and maps their contents into structured Excel sheets using an LLM.

## Features
- Upload PDFs and extract text with PyMuPDF
- Generates both a **Detailed Index** and a **Summarized Index**
- Exports to Excel (and ZIP for batches)

## Tech stack
Python · Streamlit · OpenAI API · PyMuPDF · pandas · openpyxl

## Setup
```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env
streamlit run app.py
```
