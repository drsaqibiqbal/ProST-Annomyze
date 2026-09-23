"""
Single source of truth for the capability comparison table shown in the
README and reproduced as Table 1 of the SoftwareX manuscript.

This is a CAPABILITY (feature-support) comparison, drawn from each tool's
own public documentation -- NOT a measured accuracy/recall benchmark. See
``docs/technical_details.md`` Section 9 for what is additionally required
before any quantitative superiority claim can be made, and re-verify every
cell against each tool's current documentation before relying on it, since
tool capabilities change over time and this table is a starting point, not
a continuously-verified claim.

Keeping this as data (rather than only prose in the manuscript) means the
manuscript table and the README table are generated from the same source
and cannot silently drift apart.
"""

from __future__ import annotations

__all__ = ["TOOLS", "CAPABILITIES", "MATRIX", "as_markdown_table"]

TOOLS: list[str] = [
    "Presidio Image Redactor",
    "deid (Vanderbilt / pydicom)",
    "CTP DicomAnonymizer",
    "prostate-deid",
]

CAPABILITIES: list[str] = [
    "DICOM metadata scrubbing (profile-based)",
    "Private (vendor) tag stripping",
    "Consistent ID pseudonymization",
    "Consistent date shifting",
    "Consistent UID regeneration",
    "Burned-in pixel text redaction (OCR+NER)",
    "Overlay-plane scrubbing",
    "Prostate zonal / lesion masking",
    "Anatomy-referenced intensity normalization",
    "Per-action confidence scoring",
    "Unified audit log across all steps",
    "Human-review dashboard for flagged items",
    "Single Compose-style pipeline API",
]

# Rows align with CAPABILITIES; columns align with TOOLS.
# "-" not addressed, "~" partial, "y" full support.
MATRIX = [
    ["-", "y", "y", "y"],   # DICOM metadata scrubbing
    ["-", "~", "y", "y"],   # Private tag stripping
    ["-", "~", "y", "y"],   # Consistent ID pseudonymization
    ["-", "~", "y", "y"],   # Consistent date shifting
    ["-", "~", "y", "y"],   # Consistent UID regeneration
    ["y", "-", "-", "y"],   # Burned-in pixel text redaction
    ["-", "-", "~", "y"],   # Overlay-plane scrubbing
    ["-", "-", "-", "y"],   # Prostate zonal / lesion masking
    ["-", "-", "-", "y"],   # Anatomy-referenced intensity normalization
    ["~", "-", "-", "y"],   # Per-action confidence scoring
    ["-", "~", "~", "y"],   # Unified audit log
    ["-", "-", "-", "y"],   # Human-review dashboard
    ["-", "-", "-", "y"],   # Single Compose-style API
]

_SYMBOL = {"y": "✓", "~": "partial", "-": "–"}


def as_markdown_table() -> str:
    header = "| Capability | " + " | ".join(TOOLS) + " |"
    sep = "|---" * (len(TOOLS) + 1) + "|"
    rows = [header, sep]
    for cap, row in zip(CAPABILITIES, MATRIX):
        cells = " | ".join(_SYMBOL[v] for v in row)
        rows.append(f"| {cap} | {cells} |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    print(as_markdown_table())
