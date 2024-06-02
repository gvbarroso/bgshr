import numpy as np
import warnings
from scipy import interpolate
import copy

from . import Util

## TODO: break apart and avoid repeat integrations and calculation, especially when
## using a DFE
def Bvals(
    xs,
    s,
    u,
    splines,
    L=None,
    rmap=None,
    bmap=None,
    elements=[],
    max_r=0.1,
    u_vals=None,
    s_vals=None,
    r_xs=None,
):
    """
    Get predicted reduction for given s-value
    """
    correction = False
    if bmap is not None:
        # adjust s in each element, rmap, and u in each element
        correction = True
        B_elem = [bmap.integrate(e[0], e[1]) / (e[1] - e[0]) for e in elements]
    else:
        B_elem = [1 for e in elements]

    if u_vals is None:
        u_vals = np.sort(list(set([k[0] for k in splines.keys()])))
    if s_vals is None:
        s_vals = np.sort(list(set([k[1] for k in splines.keys()])))

    if len(u_vals) != 1:
        raise ValueError("more than one mutation rate present")
    uL = u_vals[0]
    B0 = u / uL

    B = np.ones(len(xs))
    if len(elements) == 0:
        return B

    if L is None:
        L = max([xs[-1], elements[-1][1]])

    if rmap is None:
        warnings.warn("No rmap provided, assuming r=0")
        rmap = Util.build_uniform_rmap(0)

    # map positions of given x values
    if r_xs is None:
        # this can be slow to recompute each time
        # cache and pass to recursive calls
        r_xs = np.array([rmap(x) for x in xs])
    else:
        assert len(r_xs) == len(xs)

    for i, e in enumerate(elements):
        # if we correct for local B value, update the s value in this element
        s_elem = B_elem[i] * s
        if s_elem in s_vals:
            # get lengths, midpoints, and distances for the constrained element
            mid = np.mean(e)
            L_elem = e[1] - e[0]
            r_mid = rmap(mid)
            r_dists = np.abs(r_xs - r_mid)
            # restrict to distances within the maximum recombination distance
            indexes = np.where(r_dists <= max_r)[0]
            r_dists = r_dists[indexes]
            # if we correct for local B values, update the effective length of element
            L_elem *= B_elem[i]
            fac = splines[(uL, s_elem)](r_dists) ** (B0 * L_elem)
            B[indexes] *= fac
        else:
            # call this function recursively to interpolate between the s-grid
            s0 = s_vals[np.where(s_elem > s_vals)[0][-1]]
            s1 = s_vals[np.where(s_elem < s_vals)[0][0]]
            p1 = (s_elem - s0) / (s1 - s0)
            p0 = 1 - p1
            assert 0 < p0 < 1
            fac0 = Bvals(
                xs,
                s0,
                u * p0 * B_elem[i],
                splines,
                L=L,
                rmap=rmap,
                elements=[e],
                max_r=max_r,
                u_vals=u_vals,
                s_vals=s_vals,
                r_xs=r_xs,
            )
            fac1 = Bvals(
                xs,
                s1,
                u * p1 * B_elem[i],
                splines,
                L=L,
                rmap=rmap,
                elements=[e],
                max_r=max_r,
                u_vals=u_vals,
                s_vals=s_vals,
                r_xs=r_xs,
            )
            B *= fac0 * fac1
    return B


def Bmap(
    xs,
    s,
    u,
    splines,
    L=None,
    rmap=None,
    bmap=None,
    elements=[],
    max_r=0.1,
    u_vals=None,
    s_vals=None,
):
    bs = Bvals(
        xs,
        s,
        u,
        splines,
        L=L,
        rmap=rmap,
        bmap=bmap,
        elements=elements,
        max_r=max_r,
        u_vals=u_vals,
        s_vals=s_vals,
    )
    return interpolate.CubicSpline(xs, bs, bc_type="natural")
