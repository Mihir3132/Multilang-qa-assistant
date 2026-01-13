import os
import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from chunking import chunk_pages
from pdf_ingestion import extract_pdf_content, iter_page_texts
from qa_chain import ask_question, build_qa_chain
from vectorstore import build_documents, create_or_load_vectorstore


st.set_page_config(page_title="Multilingual PDF QA (Hindi/English/Hinglish)", page_icon="📄")


def ensure_tesseract():
    """
    If TESSERACT_CMD is provided, set it so pytesseract can find the binary.
    """
    if settings.tesseract_cmd:
        os.environ["TESSERACT_CMD"] = settings.tesseract_cmd


def collection_name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    safe = "".join(ch for ch in stem if ch.isalnum() or ch in "-_").strip()
    return safe or "default_pdf_collection"


def process_pdf(file_bytes: bytes, filename: str):
    """
    End-to-end ingestion: save temp PDF, extract text/OCR, chunk, embed, and build retriever.
    """
    ensure_tesseract()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Check Tesseract availability before processing
        from pdf_ingestion import _check_tesseract_available
        tesseract_available = _check_tesseract_available()
        
        pages = extract_pdf_content(tmp_path)
        page_texts = list(iter_page_texts(pages))
        
        if not page_texts:
            # Provide detailed error message
            error_msg = "No text content extracted from PDF.\n\n"
            if not tesseract_available:
                error_msg += "Tesseract OCR is not installed or not found. "
                error_msg += "For scanned PDFs, you need to:\n"
                error_msg += "1. Install Tesseract OCR\n"
                error_msg += "2. Set TESSERACT_CMD in .env file pointing to tesseract.exe\n"
                error_msg += "3. Ensure Hindi language data (hin.traineddata) is installed\n\n"
            else:
                error_msg += "The PDF may be:\n"
                error_msg += "- Corrupted or password-protected\n"
                error_msg += "- Completely image-based with poor quality\n"
                error_msg += "- Empty or unreadable\n\n"
            error_msg += "Please check the PDF and try again, or ensure Tesseract OCR is properly configured."
            raise ValueError(error_msg)
        
        chunks = chunk_pages(page_texts)
        docs = build_documents(chunks)

        collection_name = collection_name_from_filename(filename)
        vectorstore = create_or_load_vectorstore(collection_name=collection_name, docs=docs)
        qa = build_qa_chain(vectorstore)
    finally:
        # Clean up the temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return qa


def main():
    st.title("📄 Multilingual PDF QA (Hindi / English / Hinglish)")
    st.caption(
        "Upload a PDF (Hindi/English/mixed). Tables in text or images are handled via PyMuPDF + Tesseract OCR (hin+eng)."
    )

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded is not None:
        with st.spinner("Indexing PDF (text + OCR + embeddings)..."):
            try:
                qa_chain = process_pdf(uploaded.read(), uploaded.name)
                st.session_state.qa_chain = qa_chain
                st.success("PDF indexed. Ask your question below.")
            except ValueError as e:
                st.error(f"Error processing PDF: {e}")
                # Check if Tesseract is available
                from pdf_ingestion import _check_tesseract_available
                if not _check_tesseract_available():
                    with st.expander("Tesseract OCR Setup Help"):
                        st.markdown("""
                        **To enable OCR for scanned PDFs:**
                        1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
                        2. Download Hindi language data (hin.traineddata) if needed
                        3. Set `TESSERACT_CMD` in your `.env` file:
                           ```
                           TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
                           ```
                        4. Restart the application
                        """)
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                st.info("Tip: If this is a scanned PDF, ensure Tesseract OCR is installed and configured.")

    question = st.text_input("Ask a question (English / Hindi / Hinglish):")
    ask_button = st.button("Answer", type="primary")

    if ask_button:
        if not st.session_state.qa_chain:
            st.error("Please upload and index a PDF first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            # Detect language for display
            from qa_chain import detect_language
            detected_lang = detect_language(question.strip())
            lang_display = {"hindi": "हिंदी", "english": "English", "hinglish": "Hinglish"}.get(detected_lang, detected_lang)
            
            with st.spinner(f"Thinking... (Detected language: {lang_display})"):
                result = ask_question(st.session_state.qa_chain, question.strip())
            st.markdown("**Answer:**")
            st.write(result.get("answer", ""))

            sources = result.get("sources", [])
            if sources:
                st.markdown("**Sources:**")
                for idx, doc in enumerate(sources, start=1):
                    page = doc.metadata.get("page_number", "?")
                    chunk_id = doc.metadata.get("chunk_id", "?")
                    st.write(f"Source {idx}: page {page}, chunk {chunk_id}")
                    st.caption(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))


if __name__ == "__main__":
    main()

