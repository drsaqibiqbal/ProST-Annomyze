"""
Export a review bundle: every ``flagged_for_review`` audit entry, plus a
cropped thumbnail of the affected region when a bounding box is available,
for the reviewer QA dashboard (``dashboard/review_app.py``).

Rationale (see README / critical-considerations notes): ``flagged_for_review``
sitting only inside an audit-trail JSON is a dead-end data point with no
consumer. A flag nobody looks at manufactures false confidence -- worse than
not flagging at all. This module is the minimal bridge from "flagged" to
"something a human can actually triage in under a second per item."
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .core import Sample

__all__ = ["export_review_bundle"]

_BBOX_RE = re.compile(r"bbox=\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")


def _parse_bbox(detail: str) -> tuple | None:
    m = _BBOX_RE.search(detail)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def export_review_bundle(
    sample: Sample,
    out_dir: str | os.PathLike,
    study_id: str = "unknown_study",
) -> Path:
    """Write a review bundle directory: manifest.json + one thumbnail per flag.

    Parameters
    ----------
    sample:
        A processed :class:`~prostate_deid.core.Sample` (post-pipeline;
        typically the ``sample.image`` here is already de-identified, so
        thumbnails are safe to hand to a reviewer without further redaction).
    out_dir:
        Directory to write the bundle into; created if missing.
    study_id:
        Identifier embedded in the manifest so multiple studies' bundles
        can be told apart in the reviewer dashboard.

    Returns
    -------
    Path to ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flagged = [e for e in sample.audit if e.flagged_for_review]
    manifest: list[dict] = []

    for i, entry in enumerate(flagged):
        item = {
            "index": i,
            "study_id": study_id,
            "step": entry.step,
            "action": entry.action,
            "detail": entry.detail,
            "confidence": entry.confidence,
            "thumbnail": None,
        }
        bbox = _parse_bbox(entry.detail)
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            crop = np.asarray(sample.image)[..., y0:y1, x0:x1]
            thumb_path = out_dir / f"flag_{i:04d}.npy"
            np.save(thumb_path, crop)
            item["thumbnail"] = thumb_path.name
        manifest.append(item)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir
