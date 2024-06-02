import pytest

import bgshr


def test_uniform_rmap():
    L = 100
    for r in [0, 1e-8, 1.5e-6]:
        rmap = bgshr.Util.build_uniform_rmap(r, L)
        assert rmap(L) == r * L
        assert rmap(L / 2) == r * L / 2
        assert rmap(0) == 0
