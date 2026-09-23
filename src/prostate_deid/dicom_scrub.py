"""
Deterministic DICOM header scrubbing.

Policy is encoded here as named, versioned profiles; this module does not
implement any probabilistic detection -- if a tag is in the profile (or is a
private tag, by default), it is removed, full stop. That determinism is the
point: metadata scrubbing is the one PHI channel where "correct" has a
verifiable, testable answer per tag.
"""

from __future__ import annotations

from .core import AuditEntry, Sample, Transform

__all__ = ["DicomTagScrub", "resolve_profile", "PROFILES"]


# ---------------------------------------------------------------------------
# Named profiles
#
# These map the HIPAA Safe Harbor identifier categories (and a minimal
# reading of the DICOM PS3.15 Basic Application Level Confidentiality
# Profile) onto standard DICOM keyword names. This is a starting recipe, not
# a certified mapping -- see README "Compliance disclaimer": running this
# module does not by itself constitute HIPAA or GDPR compliance, and the
# profile tables below should be reviewed against the current DICOM PS3.15
# standard and your institution's policy before relying on them.
# ---------------------------------------------------------------------------

PROFILES: dict[str, list[str]] = {
    "hipaa_safe_harbor": [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientBirthTime",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "PatientMotherBirthName",
        "OtherPatientIDs",
        "OtherPatientNames",
        "OtherPatientIDsSequence",
        "ReferringPhysicianName",
        "ReferringPhysicianAddress",
        "ReferringPhysicianTelephoneNumbers",
        "PerformingPhysicianName",
        "PhysiciansOfRecord",
        "OperatorsName",
        "InstitutionName",
        "InstitutionAddress",
        "InstitutionalDepartmentName",
        "StationName",
        "RequestingPhysician",
        "PersonName",
        "EthnicGroup",
        "Occupation",
        "AdditionalPatientHistory",
        "MilitaryRank",
        "MedicalRecordLocator",
        "CurrentPatientLocation",
        "PatientInstitutionResidence",
        "DeviceSerialNumber",
    ],
    "dicom_ps3.15_basic": [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientAddress",
        "OtherPatientIDs",
        "OtherPatientNames",
        "ReferringPhysicianName",
        "InstitutionName",
        "InstitutionAddress",
        "StationName",
        "DeviceSerialNumber",
    ],
}


def resolve_profile(name: str) -> list[str]:
    """Return the list of DICOM keywords a named profile removes.

    Raises ``KeyError`` for an unknown profile name rather than silently
    returning an empty list, since a typo'd profile name should not result
    in an unscrubbed file that looks scrubbed.
    """
    if name not in PROFILES:
        raise KeyError(
            f"Unknown de-identification profile {name!r}. "
            f"Available profiles: {sorted(PROFILES)}"
        )
    return list(PROFILES[name])


def _is_private_tag(keyword_or_tag: str, dicom_meta: dict) -> bool:
    """True if a metadata key represents a private (odd-group) DICOM element.

    Private tags are detected authoritatively by group number parity (DICOM
    PS3.5 Section 7.8: private groups are always odd), not by a
    keyword-dictionary guess -- vendors do not register their private tags
    in the public dictionary, so a keyword-based check would silently miss
    exactly the tags this check exists to catch.
    """
    if not (keyword_or_tag.startswith("(") and "," in keyword_or_tag):
        return False
    try:
        group_str = keyword_or_tag.strip("()").split(",")[0]
        group = int(group_str, 16)
    except (ValueError, IndexError):
        return False
    return group % 2 == 1


class DicomTagScrub(Transform):
    """Deterministic tag removal/replacement based on a named profile.

    Parameters
    ----------
    profile:
        One of :data:`PROFILES`' keys (``"hipaa_safe_harbor"`` by default),
        or any caller-registered profile name.
    extra_tags_to_remove:
        Additional keywords to remove beyond the profile.
    exclude_tags:
        Keywords to keep even though the profile would otherwise remove
        them. Use this for fields an earlier :class:`~prostate_deid.pseudonymize.Pseudonymize`
        step has already remapped to a de-identified value (PatientID,
        StudyDate, SeriesDate, the *InstanceUID fields, ...) -- see the
        pipeline ordering note in :mod:`prostate_deid.pipeline`. Without
        this, the profile deletes the pseudonymized replacement outright
        and silently undoes the remap.
    strip_private_tags:
        When ``True`` (the default), every private (odd-group) element is
        removed regardless of profile, because standard profiles only
        cover the public DICOM dictionary and vendors routinely embed PHI
        in undocumented private tags. Guessing wrong about an unknown
        private tag's content is worse than dropping it, so the default
        policy is to drop all of them.
    """

    def __init__(
        self,
        profile: str = "hipaa_safe_harbor",
        extra_tags_to_remove: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        strip_private_tags: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.profile = profile
        self.extra_tags_to_remove = extra_tags_to_remove or []
        self.exclude_tags = set(exclude_tags or [])
        self.strip_private_tags = strip_private_tags

    def apply(self, sample: Sample) -> Sample:
        if sample.dicom_meta is None:
            return sample

        tags_to_strip = set(resolve_profile(self.profile)) | set(self.extra_tags_to_remove)
        tags_to_strip -= self.exclude_tags

        for tag in list(sample.dicom_meta.keys()):
            remove = False
            reason = None

            if tag in tags_to_strip:
                remove = True
                reason = f"profile={self.profile}"
            elif (
                self.strip_private_tags
                and tag not in self.exclude_tags
                and _is_private_tag(tag, sample.dicom_meta)
            ):
                remove = True
                reason = "private_tag_policy"

            if remove:
                del sample.dicom_meta[tag]
                sample.log(
                    AuditEntry(
                        step="dicom_tag_scrub",
                        action="removed_tag",
                        detail=f"{tag} ({reason})",
                    )
                )

        return sample
