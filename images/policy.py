"""Conservative image reuse policy."""

from __future__ import annotations

import re
from typing import Any

ALLOWED_LABELS = (
    "cc by-sa",
    "cc by-nc-sa",
    "cc by-nc",
    "cc by",
    "cc0",
    "public domain",
)
THIRD_PARTY_MARKERS = (
    "reprinted",
    "reproduced",
    "with permission",
    "google earth",
    "getty",
    "alamy",
    "shutterstock",
    "third-party",
    "third party",
)
REJECTED_MARKERS = (
    "all rights reserved",
    "publisher copyright",
    "copyrighted",
)


def normalize_license(value: str | None) -> str:
    text = " ".join((value or "").strip().lower().replace("_", " ").split())
    text = re.sub(r"\bcc-by", "cc by", text)
    for marker, label in (
        ("creativecommons.org/licenses/by-nc-nd", "cc by-nc-nd"),
        ("creativecommons.org/licenses/by-nc-sa", "cc by-nc-sa"),
        ("creativecommons.org/licenses/by-nc", "cc by-nc"),
        ("creativecommons.org/licenses/by-nd", "cc by-nd"),
        ("creativecommons.org/licenses/by-sa", "cc by-sa"),
        ("creativecommons.org/licenses/by/", "cc by"),
    ):
        if marker in text:
            text = f"{label} {text}"
            break
    text = text.replace(
        "creative commons attribution-noncommercial-noderivatives",
        "cc by-nc-nd",
    )
    text = text.replace(
        "creative commons attribution-noncommercial-sharealike",
        "cc by-nc-sa",
    )
    text = text.replace("creative commons attribution-noncommercial", "cc by-nc")
    text = text.replace("creative commons attribution-noderivatives", "cc by-nd")
    text = text.replace("creative commons attribution-sharealike", "cc by-sa")
    text = text.replace("creative commons attribution", "cc by")
    text = text.replace("creative commons zero", "cc0")
    return text


def is_no_derivatives_license(value: str | None) -> bool:
    return re.search(r"\bcc\s*by(?:-[a-z]+)*-nd\b", normalize_license(value)) is not None


def format_license_label(value: str | None, license_url: str | None = None) -> str:
    license_text = normalize_license(value)
    if re.search(r"\bcc\s*by-nc-nd\b", license_text):
        label = "CC BY-NC-ND"
    elif re.search(r"\bcc\s*by-nd\b", license_text):
        label = "CC BY-ND"
    elif re.search(r"\bcc\s*by-nc-sa\b", license_text):
        label = "CC BY-NC-SA"
    elif re.search(r"\bcc\s*by-sa\b", license_text):
        label = "CC BY-SA"
    elif re.search(r"\bcc\s*by-nc\b", license_text):
        label = "CC BY-NC"
    elif re.search(r"\bcc\s*by\b", license_text):
        label = "CC BY"
    elif "cc0" in license_text:
        label = "CC0"
    elif "public domain" in license_text:
        label = "Public Domain"
    else:
        return ""

    version = ""
    for source in (value or "", license_url or ""):
        match = re.search(r"(?<!\d)(\d+\.\d+)(?!\d)", source)
        if match:
            version = match.group(1)
            break
    return f"{label} {version}" if version else label


def assess_image(
    license_value: str | None,
    caption: str | None = None,
    credit: str | None = None,
    *,
    allow_no_derivatives: bool = False,
) -> tuple[bool, str]:
    license_text = normalize_license(license_value)
    context = " ".join((caption or "", credit or "")).lower()

    third_party = next((marker for marker in THIRD_PARTY_MARKERS if marker in context), None)
    if third_party:
        return False, f"third-party marker: {third_party}"

    rejected = next((marker for marker in REJECTED_MARKERS if marker in license_text), None)
    if rejected:
        return False, f"license rejected: {rejected}"

    if is_no_derivatives_license(license_value):
        if allow_no_derivatives:
            label = format_license_label(license_value) or "NoDerivatives"
            return True, f"{label}; unmodified complete figure only"
        return False, "no-derivatives license"

    if "cc0" in license_text:
        return True, "CC0"
    if "public domain" in license_text:
        return True, "Public Domain"
    if re.search(r"\bcc\s*by-nc-sa\b", license_text):
        return True, "CC BY-NC-SA"
    if re.search(r"\bcc\s*by-sa\b", license_text):
        return True, "CC BY-SA"
    if re.search(r"\bcc\s*by-nc\b", license_text):
        return True, "CC BY-NC"
    if re.search(r"\bcc\s*by\b", license_text):
        return True, "CC BY"
    return False, "unknown or non-reusable license"


def apply_policy(
    image: dict[str, Any],
    *,
    allow_no_derivatives: bool = False,
) -> dict[str, Any]:
    license_value = str(image.get("license") or "")
    publishable, reason = assess_image(
        license_value,
        str(image.get("caption") or ""),
        str(image.get("credit") or ""),
        allow_no_derivatives=allow_no_derivatives,
    )
    no_derivatives = is_no_derivatives_license(license_value)
    result = dict(image)
    result["publishable"] = publishable
    result["reason"] = reason
    result["derivatives_allowed"] = not no_derivatives
    result["cover_eligible"] = publishable and not no_derivatives
    return result
