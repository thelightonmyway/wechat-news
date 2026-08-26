"""PyMuPDF4LLM fallback for extracting numbered figures from paper PDFs."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import pymupdf4llm
from bs4 import BeautifulSoup

from images.policy import apply_policy, is_no_derivatives_license

USER_AGENT = "Mozilla/5.0 (compatible; wechat-news/0.1; +local-research-bot)"
FIGURE_NUMBER = re.compile(r"^\s*(?:fig(?:ure)?\.?)\s*(\d+)\s*(?:[|:.-]\s*)?", re.IGNORECASE)
PDF_EXCLUSIONS = ("supplement", "moesm", "peer-review", "peer_review", "reviewer")
CREDIT_MARKERS = (
    "credit",
    "copyright",
    "©",
    "courtesy",
    "reproduced",
    "with permission",
    "base map",
    "imagery",
)


def _canonical_license(value: str | None) -> str:
    text = " ".join((value or "").strip().lower().replace("_", " ").split())
    if "creativecommons.org/publicdomain/zero" in text or text in {"cc0", "cc-0"}:
        return "CC0"
    if "creativecommons.org/licenses/by-sa" in text or text in {"cc-by-sa", "cc by-sa"}:
        return "CC BY-SA"
    if "creativecommons.org/licenses/by/" in text or text in {"cc-by", "cc by"}:
        return "CC BY"
    if "public domain" in text:
        return "Public Domain"
    return value or ""


def _page_license(soup: BeautifulSoup, fallback: str = "") -> tuple[str, str]:
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if "creativecommons.org/" in href.lower():
            return _canonical_license(href), href
    return _canonical_license(fallback), ""


def discover_pdf_source(
    article_url: str,
    doi: str = "",
    article_license: str = "",
) -> dict[str, str]:
    """Find a formal/reference PDF and explicit article license from its landing page."""
    direct_pdf = article_url if urlparse(article_url).path.lower().endswith(".pdf") else ""
    landing_candidates: list[str] = []
    if doi:
        landing_candidates.append(f"https://doi.org/{doi.strip()}")
    if article_url and not direct_pdf:
        landing_candidates.append(article_url)

    candidates: list[tuple[int, str]] = []
    if direct_pdf:
        candidates.append((20, direct_pdf))
    landing_url = ""
    resolved_license = _canonical_license(article_license)
    license_url = ""

    with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
        for landing_candidate in dict.fromkeys(landing_candidates):
            try:
                response = client.get(
                    landing_candidate,
                    headers={"User-Agent": USER_AGENT},
                )
                response.raise_for_status()
            except Exception:
                continue
            content_type = str(response.headers.get("content-type") or "").lower()
            if "pdf" in content_type:
                candidates.append((20, str(response.url)))
                continue
            landing_url = str(response.url)
            soup = BeautifulSoup(response.text, "html.parser")
            page_license, page_license_url = _page_license(soup, resolved_license)
            if page_license:
                resolved_license = page_license
                license_url = page_license_url

            for meta in soup.find_all("meta"):
                name = str(meta.get("name") or meta.get("property") or "").lower()
                content = str(meta.get("content") or "").strip()
                if content and ("pdf" in name or urlparse(content).path.lower().endswith(".pdf")):
                    candidates.append((10, urljoin(landing_url, content)))
            for anchor in soup.find_all("a", href=True):
                href = urljoin(landing_url, str(anchor.get("href") or "").strip())
                path = urlparse(href).path.lower()
                if not path.endswith(".pdf"):
                    continue
                priority = 0 if path.endswith("_reference.pdf") else 10
                candidates.append((priority, href))
            break

    usable: list[tuple[int, str]] = []
    for priority, candidate in candidates:
        lower = candidate.lower()
        if any(marker in lower for marker in PDF_EXCLUSIONS):
            continue
        usable.append((priority, candidate))
    usable.sort(key=lambda item: item[0])
    return {
        "pdf_url": usable[0][1] if usable else "",
        "landing_url": landing_url or article_url,
        "license": resolved_license,
        "license_url": license_url,
    }


def _box_text(box: dict[str, Any]) -> str:
    return " ".join(
        " ".join(str(span.get("text") or "") for span in (line.get("spans") or []))
        for line in (box.get("textlines") or [])
    ).strip()


def _bbox(box: dict[str, Any]) -> list[float]:
    return [float(box[key]) for key in ("x0", "y0", "x1", "y1")]


def _axis_gap(first: list[float], second: list[float]) -> tuple[float, float]:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0.0)
    vertical = max(first[1] - second[3], second[1] - first[3], 0.0)
    return horizontal, vertical


def _adjacent(picture: list[float], caption: list[float]) -> bool:
    horizontal, vertical = _axis_gap(picture, caption)
    x_overlap = min(picture[2], caption[2]) - max(picture[0], caption[0])
    y_overlap = min(picture[3], caption[3]) - max(picture[1], caption[1])
    return (x_overlap > 0 and vertical <= 72.0) or (y_overlap > 0 and horizontal <= 36.0)


def _caption_continuations(
    boxes: list[dict[str, Any]],
    anchor_index: int,
    picture_index: int,
) -> list[int]:
    anchor = boxes[anchor_index]
    anchor_bbox = _bbox(anchor)
    selected = [anchor_index]
    for index, candidate in enumerate(boxes):
        if index in {anchor_index, picture_index}:
            continue
        text = _box_text(candidate)
        if not text or candidate.get("boxclass") not in {"caption", "text"}:
            continue
        candidate_bbox = _bbox(candidate)
        horizontal, _ = _axis_gap(anchor_bbox, candidate_bbox)
        same_row = abs(candidate_bbox[1] - anchor_bbox[1]) <= 8.0
        if same_row and horizontal <= 36.0:
            selected.append(index)
    return sorted(selected, key=lambda index: boxes[index]["x0"])


def _credit_from_caption(caption: str) -> str:
    protected = caption
    for abbreviation in ("Fig.", "Figs.", "s.d.", "e.g.", "i.e.", "et al."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<dot>"))
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return " ".join(
        sentence.replace("<dot>", ".").strip()
        for sentence in sentences
        if any(marker in sentence.lower() for marker in CREDIT_MARKERS)
    )


def _download_pdf(url: str, destination: Path) -> None:
    with httpx.Client(timeout=60.0, follow_redirects=True, trust_env=True) as client:
        response = client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        content = response.content
    if not content.startswith(b"%PDF"):
        raise ValueError("downloaded content is not a PDF")
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("PDF exceeds 50 MiB")
    destination.write_bytes(content)


def extract_pdf_figures(
    pdf_url: str,
    output_dir: Path,
    *,
    article_url: str = "",
    article_license: str = "",
    license_url: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract complete numbered figures without inventing a separate crop algorithm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "pdf_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir.parent / "source_reference.pdf"
    _download_pdf(pdf_url, pdf_path)

    layout = pymupdf4llm.to_json(
        pdf_path,
        write_images=True,
        image_path=str(raw_dir),
        image_format="png",
        image_dpi=200,
        force_text=True,
        show_progress=False,
    )
    if isinstance(layout, str):
        layout = json.loads(layout)
    (output_dir.parent / "pdf_layout.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    matched: list[dict[str, Any]] = []
    picture_count = 0
    rejected: list[dict[str, Any]] = []
    used_numbers: set[int] = set()
    for page in layout.get("pages", []):
        boxes = page.get("boxes", [])
        anchors: list[tuple[int, int, list[float]]] = []
        for box_index, box in enumerate(boxes):
            match = FIGURE_NUMBER.match(_box_text(box))
            if match:
                anchors.append((int(match.group(1)), box_index, _bbox(box)))

        for picture_index, picture in enumerate(boxes):
            if picture.get("boxclass") != "picture":
                continue
            picture_count += 1
            picture_bbox = _bbox(picture)
            width = picture_bbox[2] - picture_bbox[0]
            height = picture_bbox[3] - picture_bbox[1]
            if picture.get("table") or width < 120.0 or height < 80.0:
                rejected.append(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": picture_index,
                        "picture_bbox": picture_bbox,
                        "reason": "small, thin, or table region",
                    }
                )
                continue

            nearby = [
                (number, anchor_index, anchor_bbox)
                for number, anchor_index, anchor_bbox in anchors
                if number not in used_numbers and _adjacent(picture_bbox, anchor_bbox)
            ]
            if not nearby:
                rejected.append(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": picture_index,
                        "picture_bbox": picture_bbox,
                        "reason": "no adjacent Fig. N caption",
                    }
                )
                continue
            number, anchor_index, anchor_bbox = min(
                nearby,
                key=lambda value: sum(_axis_gap(picture_bbox, value[2])),
            )
            if is_no_derivatives_license(article_license):
                adjacent_pictures = [
                    index
                    for index, candidate in enumerate(boxes)
                    if candidate.get("boxclass") == "picture"
                    and _adjacent(_bbox(candidate), anchor_bbox)
                ]
                if len(adjacent_pictures) != 1:
                    rejected.append(
                        {
                            "page": page.get("page_number"),
                            "picture_box_index": picture_index,
                            "picture_bbox": picture_bbox,
                            "reason": (
                                "ND figure has multiple picture regions; complete unmodified "
                                "figure cannot be guaranteed"
                            ),
                        }
                    )
                    continue
            caption_indices = _caption_continuations(boxes, anchor_index, picture_index)
            caption = " ".join(_box_text(boxes[index]) for index in caption_indices).strip()
            if not FIGURE_NUMBER.match(caption):
                rejected.append(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": picture_index,
                        "picture_bbox": picture_bbox,
                        "reason": "caption number mismatch",
                    }
                )
                continue

            raw_path = raw_dir / f"{pdf_path.name}-{int(page['page_number']):04d}-{picture_index:02d}.png"
            if not raw_path.is_file():
                rejected.append(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": picture_index,
                        "picture_bbox": picture_bbox,
                        "reason": "PyMuPDF4LLM image output missing",
                    }
                )
                continue
            final_path = output_dir / f"figure-{number:02d}.png"
            shutil.copyfile(raw_path, final_path)
            credit = _credit_from_caption(caption)
            record = apply_policy(
                {
                    "url": f"{pdf_url}#page={page['page_number']}&figure={number}",
                    "source_url": pdf_url,
                    "article_url": article_url,
                    "local_path": str(final_path),
                    "caption": caption,
                    "original_caption": caption,
                    "alt": caption,
                    "credit": credit,
                    "license": _canonical_license(article_license),
                    "license_url": license_url,
                    "provider": "PDF Figure",
                    "image_source": "pdf_figure",
                    "image_role": "figure",
                    "metadata_title": f"Figure {number}",
                    "figure_number": number,
                    "page": int(page["page_number"]),
                    "picture_bbox": picture_bbox,
                    "caption_bboxes": [_bbox(boxes[index]) for index in caption_indices],
                    "caption_boxclasses": [boxes[index].get("boxclass") for index in caption_indices],
                },
                allow_no_derivatives=True,
            )
            matched.append(record)
            used_numbers.add(number)

    matched.sort(key=lambda image: int(image.get("figure_number") or 0))
    metadata = {
        "pdf_url": pdf_url,
        "local_pdf": str(pdf_path),
        "layout_picture_regions": picture_count,
        "matched_figures": len(matched),
        "rejected_picture_regions": rejected,
        "figures": matched,
    }
    (output_dir.parent / "pdf_figures.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return matched, metadata
