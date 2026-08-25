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
    "cc by-nd",
    "by-nd",
    "-nd",
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


def assess_image(
    license_value: str | None,
    caption: str | None = None,
    credit: str | None = None,
) -> tuple[bool, str]:
    license_text = normalize_license(license_value)
    context = " ".join((caption or "", credit or "")).lower()

    third_party = next((marker for marker in THIRD_PARTY_MARKERS if marker in context), None)
    if third_party:
        return False, f"third-party marker: {third_party}"

    rejected = next((marker for marker in REJECTED_MARKERS if marker in license_text), None)
    if rejected:
        return False, f"license rejected: {rejected}"

    if re.search(r"\bcc\s*by(?:-[a-z]+)*-nd\b", license_text):
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


def apply_policy(image: dict[str, Any]) -> dict[str, Any]:
    publishable, reason = assess_image(
        str(image.get("license") or ""),
        str(image.get("caption") or ""),
        str(image.get("credit") or ""),
    )
    result = dict(image)
    result["publishable"] = publishable
    result["reason"] = reason
    return result
