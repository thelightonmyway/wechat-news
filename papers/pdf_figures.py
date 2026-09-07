"""PyMuPDF4LLM fallback for extracting numbered figures from paper PDFs."""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx
import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; wechat-news/0.1; +local-research-bot)"
WILEY_TDM_ENDPOINT = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
FIGURE_NUMBER = re.compile(r"^\s*(?:fig(?:ure)?\.?)\s*(\d+)\s*(?:[|:.-]\s*)?", re.IGNORECASE)
PDF_EXCLUSIONS = ("supplement", "moesm", "peer-review", "peer_review", "reviewer")
WILEY_LIBRARY_HOST_SUFFIX = ".onlinelibrary.wiley.com"
def _is_wiley_library_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "onlinelibrary.wiley.com" or host.endswith(WILEY_LIBRARY_HOST_SUFFIX)


def _is_pdf_candidate_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or (
        _is_wiley_library_url(url) and path.startswith("/doi/pdf/")
    )


def discover_pdf_source(
    article_url: str,
    doi: str = "",
) -> dict[str, str]:
    """Find a formal/reference PDF from a landing page."""
    direct_pdf = article_url if urlparse(article_url).path.lower().endswith(".pdf") else ""
    landing_candidates: list[str] = []
    if doi:
        landing_candidates.append(f"https://doi.org/{doi.strip()}")
    if article_url and not direct_pdf:
        landing_candidates.append(article_url)

    candidates: list[tuple[int, str]] = []
    if direct_pdf:
        candidates.append((20, direct_pdf))
    if doi and _is_wiley_library_url(article_url):
        parsed_article = urlparse(article_url)
        candidates.append(
            (
                20,
                f"{parsed_article.scheme}://{parsed_article.netloc}/doi/pdf/{doi.strip()}",
            )
        )
    landing_url = ""

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

            for meta in soup.find_all("meta"):
                name = str(meta.get("name") or meta.get("property") or "").lower()
                content = str(meta.get("content") or "").strip()
                if content and ("pdf" in name or urlparse(content).path.lower().endswith(".pdf")):
                    candidates.append((10, urljoin(landing_url, content)))
            for anchor in soup.find_all("a", href=True):
                href = urljoin(landing_url, str(anchor.get("href") or "").strip())
                path = urlparse(href).path.lower()
                if not _is_pdf_candidate_url(href):
                    continue
                priority = 0 if path.endswith("_reference.pdf") else 10
                candidates.append((priority, href))
            if _is_wiley_library_url(landing_url) and doi:
                parsed_landing = urlparse(landing_url)
                candidates.append(
                    (
                        20,
                        f"{parsed_landing.scheme}://{parsed_landing.netloc}/doi/pdf/{doi.strip()}",
                    )
                )
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


def _save_validated_pdf(content: bytes, destination: Path) -> None:
    if not content.startswith(b"%PDF"):
        raise ValueError("downloaded content is not a PDF")
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("PDF exceeds 50 MiB")
    destination.write_bytes(content)


def _download_pdf(url: str, destination: Path) -> None:
    with httpx.Client(timeout=60.0, follow_redirects=True, trust_env=True) as client:
        response = client.get(url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        content = response.content
    _save_validated_pdf(content, destination)


def _download_wiley_tdm_pdf(
    doi: str,
    token: str,
    destination: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": "wiley_tdm",
        "attempted": True,
        "success": False,
        "status": None,
        "error": "",
    }
    endpoint = f"{WILEY_TDM_ENDPOINT}{quote(doi.strip(), safe='')}"
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True, trust_env=True) as client:
            for attempt in range(2):
                response = client.get(
                    endpoint,
                    headers={"Wiley-TDM-Client-Token": token},
                )
                result["status"] = response.status_code
                if response.status_code in {502, 503, 504} and attempt == 0:
                    time.sleep(1)
                    continue
                if response.status_code != 200:
                    result["error"] = f"HTTP {response.status_code}"
                    return result
                _save_validated_pdf(response.content, destination)
                result["success"] = True
                return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return result


def download_pdf_with_wiley_tdm(
    url: str,
    destination: Path,
    *,
    doi: str = "",
    token: str = "",
    article_url: str = "",
) -> dict[str, Any]:
    """Download an ordinary PDF, then use Wiley TDM only for Wiley failures."""
    ordinary_error = ""
    try:
        _download_pdf(url, destination)
        return {
            "source": "ordinary_pdf",
            "attempted": False,
            "success": True,
            "status": 200,
            "error": "",
        }
    except Exception as exc:
        ordinary_error = f"{type(exc).__name__}: {exc}"[:1000]

    if not token or not doi or not (
        _is_wiley_library_url(url) or _is_wiley_library_url(article_url)
    ):
        return {
            "source": "ordinary_pdf",
            "attempted": False,
            "success": False,
            "status": None,
            "error": ordinary_error,
        }

    tdm_result = _download_wiley_tdm_pdf(doi, token, destination)
    if not tdm_result["success"] and ordinary_error:
        tdm_result["ordinary_error"] = ordinary_error
    return tdm_result


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _expanded_crop_rect(bbox: list[float], page_rect: pymupdf.Rect) -> pymupdf.Rect:
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    left = max(10.0, width * 0.03)
    right = max(10.0, width * 0.03)
    top = max(24.0, height * 0.10)
    bottom = max(14.0, height * 0.05)
    return pymupdf.Rect(
        max(page_rect.x0, bbox[0] - left),
        max(page_rect.y0, bbox[1] - top),
        min(page_rect.x1, bbox[2] + right),
        min(page_rect.y1, bbox[3] + bottom),
    )


def _rect_gap(first: pymupdf.Rect, second: pymupdf.Rect) -> tuple[float, float]:
    horizontal = max(first.x0 - second.x1, second.x0 - first.x1, 0.0)
    vertical = max(first.y0 - second.y1, second.y0 - first.y1, 0.0)
    return horizontal, vertical


def _rect_overlaps_any(rect: pymupdf.Rect, candidates: list[pymupdf.Rect]) -> bool:
    return any(rect.intersects(candidate) for candidate in candidates)


def _refine_figure_crop_bounds(
    page: pymupdf.Page,
    initial_bbox: list[float],
    caption_bboxes: list[list[float]] | None = None,
) -> pymupdf.Rect:
    """Use native PDF content to recover labels without swallowing page text."""
    page_rect = page.rect
    fallback = _expanded_crop_rect(initial_bbox, page_rect)
    initial = pymupdf.Rect(*initial_bbox)
    caption_rects = [pymupdf.Rect(*bbox) for bbox in caption_bboxes or []]
    content_rect = pymupdf.Rect(initial)
    try:
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            rect = pymupdf.Rect(*block[:4])
            text = re.sub(r"\s+", " ", str(block[4] or "")).strip()
            if not text or _rect_overlaps_any(rect, caption_rects):
                continue
            horizontal_overlap = max(0.0, min(initial.x1, rect.x1) - max(initial.x0, rect.x0))
            vertical_overlap = max(0.0, min(initial.y1, rect.y1) - max(initial.y0, rect.y0))
            horizontal_gap, vertical_gap = _rect_gap(initial, rect)
            short_label = len(text) <= 120 and text.count("\n") <= 3
            touches_figure = (
                (horizontal_overlap > 8.0 and vertical_gap <= 22.0)
                or (vertical_overlap > 8.0 and horizontal_gap <= 22.0)
                or (horizontal_overlap > 0.0 and vertical_overlap > 0.0)
            )
            if short_label and touches_figure:
                content_rect |= rect

        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is None:
                continue
            drawing_rect = pymupdf.Rect(rect)
            if _rect_overlaps_any(drawing_rect, caption_rects):
                continue
            horizontal_gap, vertical_gap = _rect_gap(initial, drawing_rect)
            horizontal_overlap = max(0.0, min(initial.x1, drawing_rect.x1) - max(initial.x0, drawing_rect.x0))
            vertical_overlap = max(0.0, min(initial.y1, drawing_rect.y1) - max(initial.y0, drawing_rect.y0))
            overlaps = drawing_rect.intersects(initial)
            above_figure = drawing_rect.y1 <= initial.y0
            vertical_limit = 30.0 if above_figure else 42.0
            horizontal_threshold = 8.0 if above_figure else 0.0
            nearby = (
                (vertical_gap <= vertical_limit and horizontal_overlap > horizontal_threshold)
                or (horizontal_gap <= 30.0 and vertical_overlap > 8.0)
            )
            if not (overlaps or nearby):
                continue
            # Ignore page-sized rules/backgrounds that happen to sit beside the Figure.
            if not overlaps and drawing_rect.width > initial.width * 1.10:
                continue
            if drawing_rect.get_area() > max(initial.get_area() * 4.0, 25000.0):
                continue
            content_rect |= drawing_rect
    except Exception:
        return fallback

    if content_rect.is_empty or content_rect.get_area() <= 0:
        return fallback
    margin = max(3.0, min(8.0, min(content_rect.width, content_rect.height) * 0.02))
    refined = pymupdf.Rect(
        max(page_rect.x0, content_rect.x0 - margin),
        max(page_rect.y0, content_rect.y0 - margin),
        min(page_rect.x1, content_rect.x1 + margin),
        min(page_rect.y1, content_rect.y1 + margin),
    )
    if (
        refined.is_empty
        or refined.width < initial.width * 0.5
        or refined.height < initial.height * 0.5
        or refined.get_area() > fallback.get_area() * 1.35
    ):
        return fallback
    return refined


def _render_pdf_figure(
    pdf_path: Path,
    page_number: int,
    bbox: list[float],
    destination: Path,
    caption_bboxes: list[list[float]] | None = None,
) -> None:
    document = pymupdf.open(pdf_path)
    try:
        page = document[page_number - 1]
        crop_rect = _refine_figure_crop_bounds(page, bbox, caption_bboxes)
        pixmap = page.get_pixmap(
            dpi=200,
            alpha=False,
            clip=crop_rect,
        )
        pixmap.save(destination)
    finally:
        document.close()


def extract_pdf_figures(
    pdf_url: str,
    output_dir: Path,
    *,
    article_url: str = "",
    doi: str = "",
    wiley_tdm_token: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract complete numbered figures without inventing a separate crop algorithm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "pdf_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir.parent / "source_reference.pdf"
    pdf_download = download_pdf_with_wiley_tdm(
        pdf_url,
        pdf_path,
        doi=doi,
        token=wiley_tdm_token,
        article_url=article_url,
    )
    if not pdf_download["success"]:
        raise RuntimeError(str(pdf_download.get("error") or "PDF download failed"))

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

        groups: dict[tuple[int, int], list[tuple[int, list[float]]]] = {}
        unassigned: list[tuple[int, list[float]]] = []
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
                if _adjacent(picture_bbox, anchor_bbox)
            ]
            if not nearby:
                unassigned.append((picture_index, picture_bbox))
                continue
            number, anchor_index, _ = min(
                nearby,
                key=lambda value: sum(_axis_gap(picture_bbox, value[2])),
            )
            groups.setdefault((number, anchor_index), []).append((picture_index, picture_bbox))

        for picture_index, picture_bbox in unassigned:
            nearby_groups = [
                (key, group)
                for key, group in groups.items()
                if _adjacent(picture_bbox, _union_bbox([bbox for _, bbox in group]))
            ]
            if not nearby_groups:
                rejected.append(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": picture_index,
                        "picture_bbox": picture_bbox,
                        "reason": "no adjacent Fig. N caption",
                    }
                )
                continue
            key, _ = min(
                nearby_groups,
                key=lambda value: sum(
                    _axis_gap(
                        picture_bbox,
                        _union_bbox([bbox for _, bbox in value[1]]),
                    )
                ),
            )
            groups[key].append((picture_index, picture_bbox))

        for (number, anchor_index), group in sorted(
            groups.items(),
            key=lambda item: item[1][0][0],
        ):
            picture_indices = [picture_index for picture_index, _ in group]
            picture_bboxes = [picture_bbox for _, picture_bbox in group]
            if number in used_numbers:
                rejected.extend(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": picture_index,
                        "picture_bbox": picture_bbox,
                        "reason": "duplicate Figure number",
                    }
                    for picture_index, picture_bbox in group
                )
                continue
            picture_index = picture_indices[0]
            caption_indices = _caption_continuations(boxes, anchor_index, picture_index)
            caption = " ".join(_box_text(boxes[index]) for index in caption_indices).strip()
            union_bbox = _union_bbox(picture_bboxes)
            if not FIGURE_NUMBER.match(caption):
                rejected.extend(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": current_index,
                        "picture_bbox": current_bbox,
                        "reason": "caption number mismatch",
                    }
                    for current_index, current_bbox in group
                )
                continue

            raw_paths = [
                raw_dir / f"{pdf_path.name}-{int(page['page_number']):04d}-{current_index:02d}.png"
                for current_index in picture_indices
            ]
            missing = [
                (current_index, current_bbox)
                for (current_index, current_bbox), raw_path in zip(group, raw_paths)
                if not raw_path.is_file()
            ]
            if missing:
                rejected.extend(
                    {
                        "page": page.get("page_number"),
                        "picture_box_index": current_index,
                        "picture_bbox": current_bbox,
                        "reason": "PyMuPDF4LLM image output missing",
                    }
                    for current_index, current_bbox in missing
                )
                continue

            final_path = output_dir / f"figure-{number:02d}.png"
            try:
                _render_pdf_figure(
                    pdf_path,
                    int(page["page_number"]),
                    union_bbox,
                    final_path,
                    [_bbox(boxes[index]) for index in caption_indices],
                )
            except Exception as exc:
                if len(group) == 1:
                    try:
                        shutil.copyfile(raw_paths[0], final_path)
                    except Exception:
                        rejected.extend(
                            {
                                "page": page.get("page_number"),
                                "picture_box_index": current_index,
                                "picture_bbox": current_bbox,
                                "reason": f"complete Figure render failed: {type(exc).__name__}: {exc}",
                            }
                            for current_index, current_bbox in group
                        )
                        continue
                else:
                    rejected.extend(
                        {
                            "page": page.get("page_number"),
                            "picture_box_index": current_index,
                            "picture_bbox": current_bbox,
                            "reason": f"complete Figure render failed: {type(exc).__name__}: {exc}",
                        }
                        for current_index, current_bbox in group
                    )
                    continue

            figure_metadata: dict[str, Any] = {
                "url": f"{pdf_url}#page={page['page_number']}&figure={number}",
                "source_url": pdf_url,
                "article_url": article_url,
                "local_path": str(final_path),
                "caption": caption,
                "original_caption": caption,
                "alt": caption,
                "provider": "PDF Figure",
                "image_source": "pdf_figure",
                "image_role": "figure",
                "metadata_title": f"Figure {number}",
                "figure_number": number,
                "page": int(page["page_number"]),
                "picture_bbox": union_bbox,
                "caption_bboxes": [_bbox(boxes[index]) for index in caption_indices],
                "caption_boxclasses": [boxes[index].get("boxclass") for index in caption_indices],
            }
            if len(group) > 1:
                figure_metadata["picture_bboxes"] = picture_bboxes
            matched.append(figure_metadata)
            used_numbers.add(number)

    matched.sort(key=lambda image: int(image.get("figure_number") or 0))
    metadata = {
        "pdf_url": pdf_url,
        "local_pdf": str(pdf_path),
        "pdf_download": pdf_download,
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
