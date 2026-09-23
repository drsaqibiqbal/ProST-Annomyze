import numpy as np

from prostate_deid.core import AuditEntry, Compose, Sample, Transform


class _AddOne(Transform):
    def apply(self, sample: Sample) -> Sample:
        sample.image = sample.image + 1
        sample.log(AuditEntry(step="test", action="add_one", detail="+1"))
        return sample


def test_transform_always_apply_runs_deterministically():
    s = Sample(image=np.zeros((1, 2, 2)))
    t = _AddOne(always_apply=True)
    out = t(s)
    assert np.all(out.image == 1)
    assert len(out.audit) == 1


def test_transform_p_zero_never_runs():
    s = Sample(image=np.zeros((1, 2, 2)))
    t = _AddOne(p=0.0)
    out = t(s)
    assert np.all(out.image == 0)
    assert len(out.audit) == 0


def test_compose_accumulates_one_shared_audit_trail():
    s = Sample(image=np.zeros((1, 2, 2)))
    pipeline = Compose([_AddOne(always_apply=True), _AddOne(always_apply=True)])
    out = pipeline(s)
    assert np.all(out.image == 2)
    assert len(out.audit) == 2
    assert all(e.step == "test" for e in out.audit)


def test_audit_entry_str_includes_flag():
    e = AuditEntry(step="s", action="a", detail="d", confidence=0.4, flagged_for_review=True)
    text = str(e)
    assert "FLAGGED" in text
    assert "0.40" in text
