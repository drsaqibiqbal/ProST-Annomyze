"""
Consistent, HMAC-derived pseudonymization of patient identifiers, dates and
UIDs -- NOT anonymization. See the module docstring below and the README
"Anonymization vs. pseudonymization" section for why that distinction is
legally meaningful (GDPR Art. 4(5), recital 26) and not cosmetic.

Longitudinal research (PSA follow-up, repeat MRI over a treatment course)
needs to know that scan A and scan B, taken months apart, belong to the same
patient, and often needs the *interval* between them even when the absolute
calendar dates must not be shared. Two consequences follow directly:

  * Patient IDs and UIDs are remapped through a *consistent* keyed function,
    not randomized per file, so the same source value always produces the
    same replacement.
  * Dates are *shifted* by a per-patient offset, not deleted, so relative
    timing survives while absolute dates do not.

The mapping is retained on purpose to make this possible. That is exactly
what makes the output pseudonymized rather than anonymized: a party holding
the mapping table and the secret key can re-identify the data. Callers are
responsible for keeping the mapping store and the secret out of the
de-identified output and out of the same repository/bucket as the data
(see :class:`MappingStore`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from .core import AuditEntry, Sample, Transform

__all__ = ["Pseudonymize", "MappingStore"]

_ENV_KEY_VAR = "PROSTATE_DEID_KEY"

_UID_FIELDS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
_DATE_FIELDS = (
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "PatientBirthDate",
)
_MAX_DATE_SHIFT_DAYS = 20 * 365  # bounded to +/- 20 years


class MappingStore:
    """A JSON-backed table of original -> pseudonymized values.

    Deliberately the simplest possible persistence layer: one JSON file,
    one dict-of-dicts (``{"patient_id": {...}, "uid": {...}, "date_offset": {...}}``).
    Swap in a database-backed implementation with the same three methods
    (``get``, ``put``, ``save``) for institutional deployments that need
    access control on the mapping itself -- which is the actual re-identification
    key and must be governed more strictly than the de-identified data.

    Parameters
    ----------
    path:
        Where the mapping JSON lives. The caller chooses this location and
        is responsible for keeping it out of the de-identified output
        directory and out of version control.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        else:
            self._data = {"patient_id": {}, "uid": {}, "date_offset": {}}

    def get(self, table: str, key: str) -> str | None:
        return self._data.get(table, {}).get(key)

    def put(self, table: str, key: str, value: str) -> None:
        self._data.setdefault(table, {})[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)


def _get_secret(secret: str | None) -> bytes:
    if secret is not None:
        return secret.encode("utf-8")
    env_secret = os.environ.get(_ENV_KEY_VAR)
    if not env_secret:
        raise RuntimeError(
            f"No pseudonymization secret provided and {_ENV_KEY_VAR} is not set. "
            "Set the environment variable or pass secret= explicitly. "
            "Never hardcode the secret in source control."
        )
    return env_secret.encode("utf-8")


def _hmac_hex(secret: bytes, value: str) -> str:
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _pseudo_patient_id(secret: bytes, patient_id: str, prefix: str) -> str:
    digest = _hmac_hex(secret, f"patient_id:{patient_id}")
    return f"{prefix}{digest[:16].upper()}"


def _pseudo_uid(secret: bytes, uid: str) -> str:
    """Deterministically regenerate a UID under the 2.25 (UUID-derived) root.

    The 2.25 root is reserved by the DICOM standard for UIDs derived from a
    128-bit integer without requiring a registered organizational root, which
    is exactly what's needed here: a UID that (a) never collides with a real
    scanner/vendor UID, (b) does not leak the original institution/device UID
    root, and (c) is reproducible from the same source UID + secret.
    """
    digest = hashlib.sha256(secret + b":uid:" + uid.encode("utf-8")).digest()
    as_int = int.from_bytes(digest[:16], byteorder="big")  # 128-bit int
    new_uid = f"2.25.{as_int}"
    return new_uid[:64]  # DICOM UIDs are capped at 64 chars


def _date_offset_days(secret: bytes, patient_id: str) -> int:
    """A per-patient offset in [-_MAX_DATE_SHIFT_DAYS, _MAX_DATE_SHIFT_DAYS]."""
    digest = hashlib.sha256(secret + b":offset:" + patient_id.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], byteorder="big")
    span = 2 * _MAX_DATE_SHIFT_DAYS
    return (raw % span) - _MAX_DATE_SHIFT_DAYS


def _shift_dicom_date(value: str, offset_days: int) -> str:
    """Shift a DICOM DA-format date (YYYYMMDD) by offset_days."""
    if not re.fullmatch(r"\d{8}", value):
        return value
    d = datetime.strptime(value, "%Y%m%d").date()
    shifted = d + timedelta(days=offset_days)
    return shifted.strftime("%Y%m%d")


class Pseudonymize(Transform):
    """Consistent per-patient ID remapping, date shifting, and UID regeneration.

    Parameters
    ----------
    mapping_store:
        Path to the JSON mapping file, or an already-constructed
        :class:`MappingStore`. Required -- there is no default location,
        deliberately, so callers must make an explicit choice about where
        the re-identification key lives.
    secret:
        HMAC secret. If omitted, read from the ``PROSTATE_DEID_KEY``
        environment variable. Never has a hardcoded default.
    pseudo_id_prefix:
        Prefix for generated pseudo patient IDs (default ``"PSD-"``).
    date_fields:
        DICOM keywords (DA VR) to shift. Defaults to the common study/series
        acquisition dates plus patient birth date.
    """

    def __init__(
        self,
        mapping_store: str | os.PathLike | MappingStore,
        secret: str | None = None,
        pseudo_id_prefix: str = "PSD-",
        date_fields: tuple | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = mapping_store if isinstance(mapping_store, MappingStore) else MappingStore(mapping_store)
        self._secret = _get_secret(secret)
        self.pseudo_id_prefix = pseudo_id_prefix
        self.date_fields = date_fields or _DATE_FIELDS

    def apply(self, sample: Sample) -> Sample:
        if sample.dicom_meta is None:
            return sample
        meta = sample.dicom_meta

        original_patient_id = meta.get("PatientID")
        if original_patient_id:
            pseudo_id = self.store.get("patient_id", original_patient_id)
            if pseudo_id is None:
                pseudo_id = _pseudo_patient_id(self._secret, original_patient_id, self.pseudo_id_prefix)
                self.store.put("patient_id", original_patient_id, pseudo_id)
            meta["PatientID"] = pseudo_id
            sample.log(AuditEntry(
                step="pseudonymize", action="remapped_patient_id",
                detail="PatientID -> consistent pseudo-ID",
            ))

            offset = _date_offset_days(self._secret, original_patient_id)
            for field_name in self.date_fields:
                if field_name in meta and meta[field_name]:
                    meta[field_name] = _shift_dicom_date(meta[field_name], offset)
                    sample.log(AuditEntry(
                        step="pseudonymize", action="shifted_date",
                        detail=f"{field_name} shifted by consistent per-patient offset",
                    ))

        for uid_field in _UID_FIELDS:
            original_uid = meta.get(uid_field)
            if original_uid:
                pseudo_uid = self.store.get("uid", original_uid)
                if pseudo_uid is None:
                    pseudo_uid = _pseudo_uid(self._secret, original_uid)
                    self.store.put("uid", original_uid, pseudo_uid)
                meta[uid_field] = pseudo_uid
                sample.log(AuditEntry(
                    step="pseudonymize", action="regenerated_uid",
                    detail=f"{uid_field} -> deterministic 2.25.* UID",
                ))

        self.store.save()
        return sample
