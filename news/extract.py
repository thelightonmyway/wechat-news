"""Article text and image metadata extraction."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import warnings
from typing import Any
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup
from PIL import Image

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="nltk is not installed.*", category=UserWarning)
    from newspaper import Article

from images.policy import apply_policy, is_no_derivatives_license
from news.feeds import extract_doi

USER_AGENT = "Mozilla/5.0 (compatible; wechat-news/0.1; +local-research-bot)"
LICENSE_META_NAMES = {
    "dc.rights",
    "dcterms.rights",
    "citation_license",
    "license",
    "copyright",
}


def _page_license(soup: BeautifulSoup) -> str:
    for meta in soup.find_all("meta"):
        name = str(meta.get("name") or meta.get("property") or "").lower()
        if name in LICENSE_META_NAMES:
            content = str(meta.get("content") or "").strip()
            if content:
                return content
    license_link = soup.find("a", rel=lambda value: value and "license" in value)
    if license_link:
        return license_link.get_text(" ", strip=True) or str(license_link.get("href") or "")
    return ""


def _credit_for_figure(figure: Any) -> str:
    for node in figure.find_all(True):
        classes = " ".join(node.get("class") or []).lower()
        if any(word in classes for word in ("credit", "copyright", "source", "attribution")):
            text = node.get_text(" ", strip=True)
            if text:
                return text
    return ""


def discover_figure_images(html: str, base_url: str, page_license: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    images: list[dict[str, Any]] = []
    figure_nodes = []
    for figure in soup.find_all("figure"):
        springer_parent = figure.find_parent(class_="c-article-section__figure")
        if springer_parent is not None and str(
            springer_parent.get("id") or ""
        ).startswith("figure-"):
            continue
        figure_nodes.append(figure)
    for node in soup.select('.c-article-section__figure[id^="figure-"]'):
        if node not in figure_nodes:
            figure_nodes.append(node)
    for figure in figure_nodes:
        figure_images = figure.find_all("img")
        if not figure_images:
            continue
        image = figure_images[0]
        wiley_figure = figure.find_parent(class_="article-section__full") is not None
        springer_figure = (
            "c-article-section__figure" in (figure.get("class") or [])
            and str(figure.get("id") or "").startswith("figure-")
        )
        if wiley_figure:
            src = str(image.get("data-lg-src") or image.get("src") or "")
            image_url_source = "data-lg-src" if image.get("data-lg-src") else "src"
            title_node = figure.select_one(".figure__title")
            caption_node = figure.select_one(".figure__caption-text")
            figure_title = title_node.get_text(" ", strip=True) if title_node else ""
            caption = caption_node.get_text(" ", strip=True) if caption_node else ""
        elif springer_figure:
            src = str(
                image.get("src")
                or image.get("data-src")
                or image.get("data-original")
                or ""
            )
            image_url_source = "src"
            title_node = figure.select_one(".c-article-section__figure-caption")
            caption_node = figure.select_one(".c-article-section__figure-description")
            figure_title = title_node.get_text(" ", strip=True) if title_node else ""
            caption = caption_node.get_text(" ", strip=True) if caption_node else ""
        else:
            src = str(
                image.get("src")
                or image.get("data-src")
                or image.get("data-original")
                or ""
            )
            image_url_source = "src"
            caption_node = figure.find("figcaption")
            figure_title = ""
            caption = caption_node.get_text(" ", strip=True) if caption_node else ""
        if not src:
            continue
        figure_number_match = re.search(
            r"\bfig(?:ure)?\.?\s*(\d+)\b",
            figure_title,
            re.IGNORECASE,
        )
        alt = str(image.get("alt") or "").strip()
        credit = _credit_for_figure(figure)
        image_url = urljoin(base_url, src)
        images.append(
            apply_policy(
                {
                    "url": image_url,
                    "original_url": image_url,
                    "source_url": base_url,
                    "local_path": "",
                    "caption": caption or alt,
                    "original_caption": caption,
                    "alt": alt,
                    "credit": credit,
                    "license": page_license,
                    "image_source": "html_figure",
                    "image_role": "figure",
                    "figure_number": (
                        int(figure_number_match.group(1)) if figure_number_match else None
                    ),
                    "figure_title": figure_title,
                    "figure_image_count": len(figure_images),
                    "image_url_source": image_url_source,
                    "metadata_title": figure_title or caption or alt,
                }
            )
        )
    return images


def extract_article(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    result.setdefault("text", "")
    result.setdefault("authors", [])
    result.setdefault("images", [])
    result.setdefault("extraction_error", "")
    url = str(item.get("url") or "")
    if not url:
        result["extraction_error"] = "missing URL"
        return result

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
            response = client.get(url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            html = response.text
    except Exception as exc:
        result["extraction_error"] = f"fetch failed: {type(exc).__name__}: {exc}"
        return result

    try:
        extracted_json = trafilatura.extract(
            html,
            url=url,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        extracted = json.loads(extracted_json) if extracted_json else {}
    except Exception as exc:
        extracted = {}
        result["extraction_error"] = f"trafilatura failed: {type(exc).__name__}: {exc}"

    text = str(extracted.get("text") or "").strip()
    result["text"] = text
    result["word_count"] = len(re.findall(r"\b[\w'-]+\b", text))
    result["title"] = str(extracted.get("title") or result.get("title") or "")
    if extracted.get("description") and not result.get("summary"):
        result["summary"] = str(extracted["description"])
    if extracted.get("author"):
        result["authors"] = [
            part.strip() for part in str(extracted["author"]).split(";") if part.strip()
        ]
    result["doi"] = result.get("doi") or extract_doi(
        str(extracted.get("id") or ""),
        str(extracted.get("url") or ""),
        html,
    )

    soup = BeautifulSoup(html, "html.parser")
    page_license = _page_license(soup)
    image_records = discover_figure_images(html, url, page_license)

    try:
        article = Article(url)
        article.download(input_html=html)
        article.parse()
        if article.authors:
            result["authors"] = list(article.authors)
        if article.publish_date and not result.get("published_at"):
            result["published_at"] = article.publish_date.isoformat()
        known_urls = {record["url"] for record in image_records}
        for image_url in [article.top_image, *(article.images or set())]:
            if not image_url or image_url in known_urls:
                continue
            image_role = "hero" if image_url == article.top_image else "article_image"
            image_records.append(
                apply_policy(
                    {
                        "url": image_url,
                        "local_path": "",
                        "caption": "",
                        "alt": "",
                        "credit": "",
                        "license": page_license,
                        "image_source": "html",
                        "image_role": image_role,
                        "metadata_title": "article hero" if image_role == "hero" else "",
                    }
                )
            )
            known_urls.add(image_url)
    except Exception as exc:
        if not result["extraction_error"]:
            result["extraction_error"] = f"newspaper4k metadata failed: {type(exc).__name__}: {exc}"

    result["images"] = image_records
    result["status"] = "extracted" if text else "metadata_only"
    return result


def download_publishable_images(
    images: list[dict[str, Any]],
    output_dir: str,
) -> list[dict[str, Any]]:
    """Download only policy-approved images and normalize them to JPEG/PNG."""
    from pathlib import Path

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0, follow_redirects=True, trust_env=True) as client:
        for image in images:
            record = dict(image)
            if not record.get("publishable") or not record.get("url"):
                updated.append(record)
                continue
            descriptor = " ".join(
                str(record.get(key) or "")
                for key in ("url", "caption", "alt", "metadata_title")
            ).lower()
            if any(
                term in descriptor
                for term in (
                    "logo",
                    "site icon",
                    "favicon",
                    "advertisement",
                    "tracking pixel",
                    "web banner",
                    "ui icon",
                    "sprite",
                )
            ):
                record["publishable"] = False
                record["reason"] = "non-content image"
                updated.append(record)
                continue
            try:
                response = client.get(str(record["url"]), headers={"User-Agent": USER_AGENT})
                if response.status_code == 403:
                    result = subprocess.run(
                        [
                            "curl",
                            "--fail",
                            "--silent",
                            "--show-error",
                            "--location",
                            "--max-time",
                            "30",
                            "--user-agent",
                            USER_AGENT,
                            str(record["url"]),
                        ],
                        capture_output=True,
                        timeout=35,
                        check=True,
                    )
                    content = result.stdout
                else:
                    response.raise_for_status()
                    content = response.content
                if len(content) > 12 * 1024 * 1024:
                    raise ValueError("image exceeds 12 MiB")
                opened = Image.open(io.BytesIO(content))
                width, height = opened.size
                aspect_ratio = width / max(height, 1)
                if width < 600 or height < 350 or aspect_ratio > 4 or aspect_ratio < 0.25:
                    raise ValueError(f"non-content image dimensions: {width}x{height}")
                digest = hashlib.sha256(str(record["url"]).encode("utf-8")).hexdigest()[:16]
                if is_no_derivatives_license(str(record.get("license") or "")):
                    extension = {
                        "GIF": "gif",
                        "JPEG": "jpg",
                        "PNG": "png",
                        "WEBP": "webp",
                    }.get(str(opened.format or "").upper())
                    if not extension:
                        raise ValueError(
                            f"ND image format cannot be preserved: {opened.format or 'unknown'}"
                        )
                    path = destination / f"{digest}.{extension}"
                    path.write_bytes(content)
                elif opened.format == "PNG":
                    path = destination / f"{digest}.png"
                    opened.save(path, format="PNG", optimize=True)
                else:
                    path = destination / f"{digest}.jpg"
                    opened.convert("RGB").save(path, format="JPEG", quality=90, optimize=True)
                record["local_path"] = str(path)
            except Exception as exc:
                record["publishable"] = False
                record["reason"] = f"download failed: {type(exc).__name__}: {exc}"
            updated.append(record)
    return updated
