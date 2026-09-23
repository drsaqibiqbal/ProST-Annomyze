import numpy as np

from prostate_deid.organ_mask import LESION, WHOLE_GLAND, OrganMask


def test_organ_mask_zeroes_outside_whole_gland(synthetic_sample):
    m = OrganMask(keep_labels=WHOLE_GLAND, always_apply=True)
    out = m(synthetic_sample)

    background = synthetic_sample.mask == 0
    assert np.all(out.image[..., background] == 0.0)

    gland = np.isin(synthetic_sample.mask, list(WHOLE_GLAND))
    assert not np.all(out.image[..., gland] == 0.0)


def test_organ_mask_can_restrict_to_lesion_only(synthetic_sample):
    m = OrganMask(keep_labels={LESION}, fill_value=-1.0, always_apply=True)
    out = m(synthetic_sample)
    non_lesion = synthetic_sample.mask != LESION
    assert np.all(out.image[..., non_lesion] == -1.0)


def test_no_mask_is_a_noop():
    from prostate_deid.core import Sample
    s = Sample(image=np.ones((1, 4, 4, 4)))
    out = OrganMask(always_apply=True)(s)
    assert np.all(out.image == 1.0)
    assert out.audit == []
