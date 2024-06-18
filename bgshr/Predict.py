import numpy as np
import warnings
from scipy import interpolate
import copy

from . import Util


def Bvals(
    xs,
    s,
    splines,
    u=1e-8,
    L=None,
    rmap=None,
    r=None,
    bmap=None,
    elements=[],
    max_r=0.1,
    r_dists=None,
    B_elem=None,
):
    """
    Get predicted reduction for given s-value, assuming a constant per-base
    deleterious mutation rate of u within elements.

    If s is a single value, we compute just a single B value reduction array
    over the xs values. If s is a list of values, we repeat for each s in that
    list, and return a list of B value recution arrays.

    The mutation rate(s) u can be a single scalar or a list the length of
    elements, in which case each element has its own average mutation rate.
    """
    # Apply interference correction if bmap is provided
    if B_elem is None:
        B_elem = _get_B_per_element(bmap, elements)

    # Get grid of s and u values
    u_vals = np.sort(list(set([k[0] for k in splines.keys()])))
    s_vals = np.sort(list(set([k[1] for k in splines.keys()])))
    if len(u_vals) != 1:
        raise ValueError("more than one mutation rate present")
    uL = u_vals[0]

    # Build recombination map if needed
    if rmap is None:
        if L is None:
            L = max([xs[-1], elements[-1][1]])
        # TODO allow for constant recombination rate, handle with bmap if provided?
        if r is None:
            warnings.warn("No recombination rates provided, assuming r=1e-8")
            r = 1e-8
        rmap = Util.build_uniform_rmap(r, L)

    ## TODO: pull the rec map adjustment back inside this function, since
    ## recursive calls should not recompute distances from the rec map

    # Allow for multiple s values to be computed, to reduce number
    # of calls to the recombination map when integrating over a DFE
    if np.isscalar(s):
        all_s = [s]
    else:
        all_s = [_ for _ in s]
    # Each element could have its own average mutation rate
    if np.isscalar(u):
        all_u = [u for _ in elements]
    else:
        all_u = [_ for _ in u]
    if not len(all_u) == len(elements):
        raise ValueError("list of mutation rates and elements must be same length")

    Bs = [np.ones(len(xs)) for s_val in all_s]
    if len(elements) > 0:
        if r_dists is None:
            # recombination distances between each position in xs and element midpoint
            r_dists = _get_r_dists(xs, elements, rmap)

        for j, s_val in enumerate(all_s):
            for i, (e, u_val) in enumerate(zip(elements, all_u)):
                # if we correct for local B value, update the s value in this element
                u_elem = B_elem[i] * u_val
                s_elem = B_elem[i] * s_val
                # scalar factor to account for differences btw input u and lookup table
                u_fac = u_elem / uL
                if s_elem in s_vals:
                    # get length of the constrained element
                    L_elem = e[1] - e[0]
                    # restrict to distances within the maximum recombination distance
                    indexes = np.where(r_dists[i] <= max_r)[0]
                    r_dist = r_dists[i][indexes]
                    fac = splines[(uL, s_elem)](r_dist) ** (u_fac * L_elem)
                    Bs[j][indexes] *= fac
                else:
                    # call this function recursively to interpolate between the s-grid
                    s0, s1, p0, p1 = _get_interpolated_svals(s_elem, s_vals)
                    # adjust mutation rates by interpolation and any local B correction
                    u0 = u_val * B_elem[i] * p0
                    u1 = u_val * B_elem[i] * p1
                    fac0 = Bvals(
                        xs,
                        s0,
                        splines,
                        u=u0,
                        L=L,
                        rmap=rmap,
                        elements=[e],
                        max_r=max_r,
                        r_dists=[r_dists[i]],
                    )
                    fac1 = Bvals(
                        xs,
                        s1,
                        splines,
                        u=u1,
                        L=L,
                        rmap=rmap,
                        elements=[e],
                        max_r=max_r,
                        r_dists=[r_dists[i]],
                    )
                    Bs[j] *= fac0 * fac1

    if np.isscalar(s):
        assert len(Bs) == 1
        return Bs[0]
    else:
        return Bs


def _get_B_per_element(bmap, elements):
    if bmap is None:
        # without bmap, correction factor is 1
        B_elem = np.ones(len(elements))
    else:
        # average B in each element
        B_elem = np.array(
            [bmap.integrate(e[0], e[1]) / (e[1] - e[0]) for e in elements]
        )
    return B_elem


def _get_element_midpoints(elements):
    x = np.zeros(len(elements))
    for i, e in enumerate(elements):
        x[i] = np.mean(e)
    return x


def _get_r_dists(xs, elements, rmap):
    # element midpoints
    x_midpoints = _get_element_midpoints(elements)
    # get recombination map values
    r_midpoints = rmap(x_midpoints)
    r_xs = rmap(xs)
    r_dists = np.zeros((len(elements), len(xs)))
    # get recombination fractions
    for i, r_mid in enumerate(r_midpoints):
        r_dists[i] = Util.haldane_map_function(np.abs(r_xs - r_mid))
    return r_dists


def _get_interpolated_svals(s_elem, s_vals):
    s0 = s_vals[np.where(s_elem > s_vals)[0][-1]]
    s1 = s_vals[np.where(s_elem < s_vals)[0][0]]
    p1 = (s_elem - s0) / (s1 - s0)
    p0 = 1 - p1
    assert 0 < p0 < 1
    return s0, s1, p0, p1


def Bmap(
    xs,
    s,
    splines,
    u=1e-8,
    L=None,
    rmap=None,
    bmap=None,
    elements=[],
    max_r=0.1,
):
    bs = Bvals(
        xs,
        s,
        splines,
        u=u,
        L=L,
        rmap=rmap,
        bmap=bmap,
        elements=elements,
        max_r=max_r,
    )
    return interpolate.CubicSpline(xs, bs, bc_type="natural")


#
# def Bvals_dfe(
#    xs,
#    ss,
#    splines,
#    u=1e-8,
#    shape=None,
#    scale=None,
#    L=None,
#    rmap=None,
#    bmap=None,
#    elements=[],
#    max_r=0.1,
# ):
#    if shape is None or scale is None:
#        raise ValueError("must provide gamma dfe shape and scale")
#
#    Bs = Bvals(
#        xs, ss, splines, u=u, L=L, rmap=rmap, bmap=bmap, elements=elements, max_r=max_r
#    )
#    B = Util.integrate_gamma_dfe(Bs, ss, shape, scale)
#    return B
