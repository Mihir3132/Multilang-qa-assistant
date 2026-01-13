from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from config import settings


logger = logging.getLogger(__name__)

# Auto-detect Tesseract if not explicitly configured
def _auto_detect_tesseract() -> str | None:
    """Try to auto-detect Tesseract executable."""
    # If explicitly set, use it
    if settings.tesseract_cmd:
        return settings.tesseract_cmd
    
    # Check common Windows locations
    import os
    possible_paths = [
        # Current directory (common for portable installations)
        os.path.join(os.getcwd(), "tesseract.exe"),
        # Common installation paths
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        # User might have it in PATH
        "tesseract.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Auto-detected Tesseract at: {path}")
            return path
    
    return None

# Set Tesseract path
_tesseract_path = _auto_detect_tesseract()
if _tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path
    logger.info(f"Using Tesseract from: {_tesseract_path}")

# Check if Tesseract is available
_tesseract_available = None


def _check_tesseract_available() -> bool:
    """Check if Tesseract OCR is available."""
    global _tesseract_available, _tesseract_path
    
    if _tesseract_available is not None:
        return _tesseract_available
    
    # Try auto-detection if not already set
    if not _tesseract_path:
        _tesseract_path = _auto_detect_tesseract()
        if _tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = _tesseract_path
    
    try:
        version = pytesseract.get_tesseract_version()
        _tesseract_available = True
        logger.info(f"Tesseract OCR is available (version: {version})")
        
        # Check for language packs
        try:
            langs = pytesseract.get_languages()
            if "hin" in langs and "eng" in langs:
                logger.info("Hindi and English language packs are available")
            elif "eng" in langs:
                logger.warning("English language pack found, but Hindi (hin) may be missing")
            else:
                logger.warning("Language packs may be missing")
        except Exception:
            logger.warning("Could not check language packs")
            
    except (pytesseract.TesseractNotFoundError, Exception) as e:
        _tesseract_available = False
        logger.warning(
            f"Tesseract OCR not found: {e}. OCR features will be disabled. "
            "Install Tesseract and set TESSERACT_CMD in .env if needed. "
            "Text-based PDFs will still work."
        )
    return _tesseract_available


@dataclass
class PageContent:
    page_number: int
    text: str


def _extract_text_from_page(page: fitz.Page) -> str:
    """Extract textual content from a PDF page using PyMuPDF."""
    text = page.get_text("text") or ""
    return text


def _is_page_scanned(page: fitz.Page, min_text_length: int = 50) -> bool:
    """
    Detect if a page is scanned (has very little extractable text).
    Returns True if the page appears to be a scanned image.
    """
    text = _extract_text_from_page(page)
    return len(text.strip()) < min_text_length


def _preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy:
    - Resize to higher DPI if needed
    - Enhance contrast
    - Apply slight sharpening
    - Convert to grayscale if needed
    """
    # Resize to higher resolution for better OCR
    if settings.ocr_image_dpi > 0:
        # Calculate scale factor (assuming original is ~72 DPI)
        scale = settings.ocr_image_dpi / 72.0
        if scale > 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to grayscale for better OCR
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Slight sharpening
    image = image.filter(ImageFilter.SHARPEN)
    
    return image


def _render_page_as_image(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """
    Render the entire PDF page as a high-resolution image.
    Useful for scanned PDFs where text extraction fails.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def _extract_images_from_page(page: fitz.Page) -> List[Image.Image]:
    """Extract images from a PDF page as PIL Images."""
    images: List[Image.Image] = []
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(image)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to decode image from page %s: %s", page.number, exc)
    return images


def _ocr_image(image: Image.Image, lang: str = "hin+eng", psm: int = None) -> str:
    """
    Run Tesseract OCR on a PIL Image with improved settings.
    Returns empty string if Tesseract is not available.
    Falls back to English-only if multi-language fails.
    
    Args:
        image: PIL Image to OCR
        lang: Language code (e.g., "hin+eng" for Hindi+English)
        psm: Page Segmentation Mode (uses settings.ocr_psm_mode if None)
    """
    if not _check_tesseract_available():
        return ""
    
    # Preprocess image for better OCR
    processed_image = _preprocess_image_for_ocr(image)
    
    # Use PSM mode from settings if not provided
    if psm is None:
        psm = settings.ocr_psm_mode
    
    config = f'--psm {psm} --oem 3'
    
    # Try with requested language first
    try:
        result = pytesseract.image_to_string(processed_image, lang=lang, config=config)
        if result and result.strip():
            return result
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract not found during OCR. Skipping image OCR.")
        return ""
    except Exception as exc:
        # If multi-language fails, try English-only as fallback
        if lang != "eng" and "+" in lang:
            logger.info(f"Multi-language OCR ({lang}) failed, trying English-only: {exc}")
            try:
                result = pytesseract.image_to_string(processed_image, lang="eng", config=config)
                if result and result.strip():
                    logger.info("English-only OCR succeeded")
                    return result
            except Exception as fallback_exc:
                logger.warning(f"English-only OCR also failed: {fallback_exc}")
        else:
            logger.warning(f"OCR failed: {exc}")
    
    return ""


def extract_pdf_content(pdf_path: str | Path) -> List[PageContent]:
    """
    Extract text (including tables as text) and OCR text from images
    for each page of the PDF.

    Enhanced for scanned PDFs:
    - Detects scanned pages (little/no text)
    - Renders entire page as image and OCRs it for scanned pages
    - Uses improved OCR settings and image preprocessing
    - Handles both text-based and image-based tables
    - Falls back to full-page OCR if text extraction yields little content
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Check Tesseract availability early
    tesseract_available = _check_tesseract_available()
    if not tesseract_available:
        logger.warning("Tesseract OCR is not available. Scanned PDFs may not be processed correctly.")

    doc = fitz.open(pdf_path)
    pages: List[PageContent] = []
    total_pages = len(doc)
    pages_with_text = 0
    pages_ocr_attempted = 0
    pages_ocr_successful = 0

    for page_index in range(total_pages):
        page = doc[page_index]
        page_number = page_index + 1

        # Text directly from PDF (including text-based tables)
        text_parts: List[str] = []
        text = _extract_text_from_page(page)
        has_substantial_text = text and len(text.strip()) > 10
        
        if has_substantial_text:
            text_parts.append(text)
            pages_with_text += 1

        # Check if page is scanned (has very little text)
        is_scanned = _is_page_scanned(page)
        
        # Try OCR if:
        # 1. Page appears scanned (little/no text), OR
        # 2. We have no substantial text (fallback)
        should_try_full_page_ocr = (is_scanned or not has_substantial_text) and tesseract_available
        
        if should_try_full_page_ocr:
            # For scanned pages or pages with no text, render entire page as image and OCR
            pages_ocr_attempted += 1
            logger.info(f"Page {page_number} needs OCR (scanned={is_scanned}, has_text={has_substantial_text}). Attempting full-page OCR.")
            try:
                page_image = _render_page_as_image(page, dpi=settings.ocr_image_dpi)
                ocr_text = _ocr_image(page_image, lang="hin+eng", psm=6)  # PSM 6 for uniform block
                if ocr_text and ocr_text.strip():
                    text_parts.append(ocr_text)
                    pages_ocr_successful += 1
                    logger.info(f"Extracted {len(ocr_text)} characters from OCR on page {page_number}")
                else:
                    logger.warning(f"OCR returned empty text for page {page_number}")
            except Exception as exc:
                logger.error(f"Failed to OCR page {page_number}: {exc}", exc_info=True)
        elif not has_substantial_text and not tesseract_available:
            # No text and no OCR available
            logger.warning(f"Page {page_number} has no extractable text and Tesseract OCR is not available.")
        
        # Also try OCR on embedded images (for hybrid PDFs)
        if has_substantial_text or not should_try_full_page_ocr:
            images = _extract_images_from_page(page)
            for image in images:
                if tesseract_available:
                    ocr_text = _ocr_image(image, lang="hin+eng")
                    if ocr_text.strip():
                        text_parts.append(ocr_text)

        combined_text = "\n".join(part.strip() for part in text_parts if part.strip())
        if not combined_text.strip():
            logger.warning(f"No text extracted from page {page_number}")
        
        pages.append(PageContent(page_number=page_number, text=combined_text))

    doc.close()
    
    # Log summary
    logger.info(
        f"PDF extraction summary: {total_pages} pages, "
        f"{pages_with_text} with text, {pages_ocr_attempted} OCR attempts, "
        f"{pages_ocr_successful} successful OCR"
    )
    
    return pages


def iter_page_texts(pages: Iterable[PageContent]) -> Iterable[tuple[str, dict]]:
    """
    Yield (text, metadata) tuples for downstream LangChain ingestion.

    Metadata includes:
      - page_number
    """
    for page in pages:
        if not page.text.strip():
            continue
        metadata = {"page_number": page.page_number}
        yield page.text, metadata

