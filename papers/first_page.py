"""Render a paper PDF first page for article display and WeChat cover use."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf


def render_paper_first_page(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 220,
) -> dict[str, Any]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"paper PDF not found: {pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)
    try:
        if document.page_count < 1:
            raise ValueError("paper PDF has no pages")
        page = document[0]
        full_path = output_dir / "paper-first-page.png"
        full_pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        full_pixmap.save(full_path)

        cover_ratio = 900 / 383
        cover_height = min(page.rect.height, page.rect.width / cover_ratio)
        cover_rect = pymupdf.Rect(0, 0, page.rect.width, cover_height)
        cover_path = output_dir / "paper-first-page-cover.png"
        cover_pixmap = page.get_pixmap(dpi=dpi, alpha=False, clip=cover_rect)
        cover_pixmap.save(cover_path)
        page_bbox = [0.0, 0.0, float(page.rect.width), float(page.rect.height)]
        cover_bbox = [
            float(cover_rect.x0),
            float(cover_rect.y0),
            float(cover_rect.x1),
            float(cover_rect.y1),
        ]
    finally:
        document.close()

    return {
        "image_source": "paper_first_page",
        "image_role": "paper_first_page",
        "source_pdf": str(pdf_path),
        "page": 1,
        "local_path": str(full_path),
        "wechat_cover_path": str(cover_path),
        "dpi": dpi,
        "page_bbox": page_bbox,
        "cover_bbox": cover_bbox,
    }
