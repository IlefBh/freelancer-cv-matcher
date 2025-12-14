from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import docx
import pdfplumber


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CV not found: {path}")

    parts = []
    with pdfplumber.open(str(p)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)

    return _clean_text("\n".join(parts))


def extract_text_from_docx(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CV not found: {path}")

    d = docx.Document(str(p))
    parts = [para.text for para in d.paragraphs if para.text and para.text.strip()]
    return _clean_text("\n".join(parts))


def extract_cv_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)

    raise ValueError("Unsupported CV format. Use .pdf or .docx")


def build_profile_text(cv_text: str) -> str:
    """
    Keep it simple: cleaned CV text is enough for TF-IDF matching.
    Later you can add skill extraction / section weighting if you want.
    """
    return _clean_text(cv_text)
