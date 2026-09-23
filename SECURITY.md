# Security Policy

`prostate-deid` handles PHI redaction. For this library, two categories of
report matter and are treated with equal urgency:

1. **Conventional security vulnerabilities** (dependency CVEs, code
   execution, path traversal in file I/O, etc.).
2. **De-identification gaps** — a case where the library fails to remove,
   or incorrectly re-identifies, protected health information. Because a
   missed detection is a privacy incident rather than a normal bug, please
   report these the same way you would a security vulnerability, **not**
   as a public GitHub issue.

## Reporting

Please email `[TODO: security-contact@example.com]` with:

- A description of the gap or vulnerability.
- Steps to reproduce, using **synthetic** data only (see `CONTRIBUTING.md`
  — never attach real patient data or a real DICOM file to a report).
- The library version and, if relevant, the DICOM transfer syntax / vendor
  involved (for private-tag or pixel-redaction gaps).

We aim to acknowledge reports within 5 business days. Please allow a
reasonable disclosure window before any public discussion of a
de-identification gap, since it may affect data already shared by other
users of the library.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x   | :white_check_mark: |

## Not a compliance guarantee

Reporting and fixing a de-identification gap does not retroactively
guarantee that data processed by a prior version was adequately
de-identified. See the README "Compliance disclaimer" — this library is a
tool toward HIPAA/GDPR compliance, not a certification of it.
