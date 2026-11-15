"""Pruebas de las particiones temporales del protocolo (05-protocolo-temporal.md)."""

import numpy as np
import pytest

from dm_credit.modeling import folds as fd


def test_fold_bounds_match_protocol():
    got = {f.name: (f.fit, f.val) for f in fd.FOLDS}
    assert got == {
        "P1": ((0, 39), (40, 49)),
        "P2": ((0, 49), (50, 59)),
        "P3": ((0, 59), (60, 73)),
    }
    assert fd.OOT_WEEKS == (74, 91)


def test_no_overlap_between_fit_and_val_and_oot_never_used():
    for f in fd.FOLDS:
        assert f.fit[1] < f.val[0]
        assert f.val[1] < fd.OOT_WEEKS[0]
        assert f.cutoff_week == f.fit[1] + 1


def test_split_masks_exact_bounds():
    weeks = np.arange(0, 92)
    f = fd.get_fold("P1")
    fit, val = fd.split_masks(weeks, f)
    assert weeks[fit].min() == 0 and weeks[fit].max() == 39
    assert weeks[val].min() == 40 and weeks[val].max() == 49
    assert not (fit & val).any()


def test_oot_mask_and_unknown_fold():
    weeks = np.arange(0, 100)
    m = fd.oot_mask(weeks)
    assert weeks[m].min() == 74 and weeks[m].max() == 91
    with pytest.raises(KeyError):
        fd.get_fold("P9")
