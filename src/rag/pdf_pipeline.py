from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import io

import fitz  # PyMuPDF
from PIL import Image

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None


@dataclass
class ContentUnit:
    text: str
    metadata: Dict[str, Any]


# Create OCR engine once (expensive to init)
_OCR: Optional["PaddleOCR"] = None

def get_ocr() -> Optional["PaddleOCR"]:
    global _OCR
    if PaddleOCR is None:
        return None
    if _OCR is None:
        # angle_cls=True helps rotated/scanned docs
        _OCR = PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR


def render_page_to_pil(pdf_path: Path, page_idx: int, dpi: int = 350) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    page = doc.load_page(page_idx)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img


def paddle_ocr_page(img: Image.Image) -> tuple[str, float]:
    """
    Returns (text, avg_confidence)
    """
    ocr = get_ocr()
    if ocr is None:
        return "", 0.0

    # PaddleOCR expects numpy array or path; PIL -> numpy
    import numpy as np
    arr = np.array(img)

    result = ocr.ocr(arr, cls=True)  # list of lines
    # result shape: [[ [box], (text, conf) ], ...]
    lines = []
    confs = []

    if not result:
        return "", 0.0

    # result can be nested (depending on version)
    page_res = result[0] if isinstance(result[0], list) and result and isinstance(result[0][0], list) else result

    for item in page_res:
        if not item or len(item) < 2:
            continue
        txt, conf = item[1]
        if txt:
            lines.append(txt)
            try:
                confs.append(float(conf))
            except Exception:
                pass

    text = "\n".join(lines).strip()
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text, avg_conf


def full_page_ocr_paddle(pdf_path: Path, dpi: int = 350, min_chars: int = 30) -> List[ContentUnit]:
    units: List[ContentUnit] = []
    ocr = get_ocr()
    if ocr is None:
        return units

    doc = fitz.open(str(pdf_path))
    for page_idx in range(doc.page_count):
        img = render_page_to_pil(pdf_path, page_idx, dpi=dpi)
        text, avg_conf = paddle_ocr_page(img)

        if text and len(text) >= min_chars:
            units.append(ContentUnit(
                text=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_idx,
                    "kind": "page_ocr",
                    "extractor": f"pymupdf_render_{dpi}dpi+paddleocr",
                    "avg_conf": round(avg_conf, 3),
                    "has_ocr_text": True,
                }
            ))
    return units

def extract_text_pymupdf(pdf_path: Path) -> List[ContentUnit]:
    units: List[ContentUnit] = []
    doc = fitz.open(str(pdf_path))
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        txt = (page.get_text("text") or "").strip()
        if txt:
            units.append(ContentUnit(
                text=txt,
                metadata={"source": str(pdf_path), "page": page_idx, "kind": "text", "extractor": "pymupdf"}
            ))
    return units


def extract_all_units(
    pdf_path: Path,
    include_text: bool = True,
    include_tables: bool = True,
    include_images: bool = False,
    ocr_pages: bool = True,
    ocr_images: Optional[bool] = None,  # ✅ alias for older callers
) -> List[ContentUnit]:
    # If caller passed ocr_images, treat it as ocr_pages
    if ocr_images is not None:
        ocr_pages = ocr_images
    
    units: List[ContentUnit] = []

    text_units = extract_text_pymupdf(pdf_path) if include_text else []
    units.extend(text_units)

    # If you already have tables extraction, keep it here
    # units.extend(extract_tables_pdfplumber(pdf_path))  # optional

    if ocr_pages:
        pages_with_text = {u.metadata["page"] for u in text_units if (u.text or "").strip()}
        # Run OCR for pages that don't have text layer
        ocr_units = full_page_ocr_paddle(pdf_path, dpi=350, min_chars=10)
        for u in ocr_units:
            if u.metadata["page"] not in pages_with_text:
                units.append(u)

    return units