"""
Utility functions for `bgshr`, including recombination map handling, reading
input files, and handling input data in lookup tables.
"""

import numpy as np
import pandas
from scipy import interpolate
from scipy import integrate
from scipy import stats


def load_lookup_table(df_name, sep=","):
    df = pandas.read_csv(df_name, sep=sep)
    return df


def subset_lookup_table(df, Na=10000, uL=1e-8, generation=0):
    df_sub = df[
        (df["Na"] == 10000) & (df["uL"] == uL) & (df["Generation"] == generation)
    ]
    return df_sub


def generate_cubic_splines(df_sub):
    """
    df_sub is the dataframe subsetted to a single Na, uR and t

    Cubic spline functions are created over r, for each combination of
    uL and s, returning the fractional reduction based on piR and pi0.

    This function returns the (sorted) arrays of uL and s and the
    dictionary of cubic spline functions with keys (uL, s).
    """
    # Check that only
    assert len(np.unique(df_sub["Na"])) == 1
    assert len(np.unique(df_sub["uR"])) == 1
    assert len(np.unique(df_sub["t"])) == 1

    # Get arrays of selection and mutation values
    s_vals = np.array(sorted(list(set(df_sub["s"]))))
    u_vals = np.array(sorted(list(set(df_sub["uL"]))))

    # Store cubic splines of fractional reduction for each pair of u and s, over r
    splines = {}
    for u in u_vals:
        for s in s_vals:
            key = (u, s)
            # Subset to given s and u values
            df_s_u = df_sub[(df_sub["s"] == s) & (df_sub["uR"] == u)]
            rs = np.array(df_s_u["r"])
            Bs = np.array(df_s_u["B"])
            splines[key] = interpolate.CubicSpline(rs, Bs, bc_type="natural")

    return u_vals, s_vals, splines


def build_uniform_rmap(r, L):
    """
    Builds a cumulative recombination map for given per-base recombination rate r.
    """
    pos = [0, L]
    cum = [0, r * L]
    return interpolate.interp1d(pos, cum)


def build_pw_constant_rmap(pos, rates):
    pass


def adjust_uniform_rmap(r, L, bmap, steps=1000):
    # this may need to be revisited
    pos = [0]
    cum = [0]
    for x in np.linspace(0, L, steps + 1)[1:]:
        pos.append(x)
        cum.append(bmap.integrate(0, x) * r)
    return interpolate.interp1d(pos, cum)


def build_recombination_map(pos, rates):
    """
    From arrays of positions and rates, build a linearly interpolated recombination
    map. The length of pos must be one greater than rates, and must be monotonically
    increasing.
    """
    cum = [0]
    for left, right, rate in zip(pos[:-1], pos[1:], rates):
        if rate < 0:
            raise ValueError("recombination rate is negative")
        bp = right - left
        if bp <= 0:
            raise ValueError("positions are not monotonically increasing")
        cum += [cum[-1] + bp * rate]
    return interpolate.interp1d(pos, cum)


def adjust_recombination_map(rmap, bmap):
    # from interp1d get the x and ys, adjust ys, and rebuild
    # probably not that nice of a way to do it...
    pos = np.sort(np.unique(np.concatenate((bmap.x, rmap.x))))
    map_pos = rmap.x
    map_vals = np.diff(rmap.y) / np.diff(rmap.x)
    bs = np.array([bmap.integrate(a, b) for a, b in zip(pos[:-1], pos[1:])]) / np.diff(
        pos
    )
    rates = np.zeros(len(pos) - 1)
    i = -1
    for j, x in enumerate(pos):
        if x in map_pos:
            if x == map_pos[-1]:
                break
            i += 1
        rates[j] = map_vals[i]
    rates *= bs
    rmap = build_recombination_map(pos, rates)
    return rmap


def load_recombination_map(fname, L=None):
    """
    Get positions and rates to build recombination map
    """
    map_df = pandas.read_csv(fname, sep="\s+")
    pos = np.concatenate(([0], map_df["Position(bp)"]))
    rates = np.concatenate(([0], map_df["Rate(cM/Mb)"])) / 100 / 1e6
    if L is not None:
        assert L > pos[-1]
        pos = np.insert(pos, len(pos), L)
    else:
        rates = rates[:-1]
    assert len(rates) == len(pos) - 1
    rmap = build_recombination_map(pos, rates)
    return rmap


def haldane_map_function(rs):
    return 0.5 * (1 - np.exp(-2 * rs))


def load_elements(bed_file):
    elem_left = []
    elem_right = []
    data = pandas.read_csv(bed_file, sep="\t", header=None)
    elem_left = np.array(data[1])
    elem_right = np.array(data[2])
    elements = np.zeros((len(elem_left), 2), dtype=int)
    elements[:, 0] = elem_left
    elements[:, 1] = elem_right
    return elements


def collapse_elements(elements):
    elements_comb = []
    for e in elements:
        if len(elements_comb) == 0:
            elements_comb.append(e)
        elif e[0] <= elements_comb[-1][1]:
            assert e[0] >= elements_comb[-1][0]
            elements_comb[-1][1] = e[1]
        else:
            elements_comb.append(e)
    return np.array(elements_comb)


def break_up_elements(elements, max_size=500):
    elements_br = []
    for l, r in elements:
        if r - l > max_size:
            num_breaks = (r - l) // max_size
            z = np.floor(np.linspace(l, r, 2 + num_breaks)).astype(int)
            for x, y in zip(z[:-1], z[1:]):
                elements_br.append([x, y])
        else:
            elements_br.append([l, r])
    return np.array(elements_br)


def weights_gamma_dfe(s_vals, shape, scale):
    assert np.all(s_vals <= 0)
    s_vals_sorted = np.sort(s_vals)
    if np.any(s_vals != s_vals_sorted):
        raise ValueError("selection values are not sorted")

    pdf = stats.gamma.pdf(-s_vals, shape, scale=scale)
    grid = np.concatenate(([s_vals[0]], s_vals, [s_vals[-1]]))
    weights = (grid[2:] - grid[:-2]) / 2 * pdf
    return weights


def integrate_with_weights(vals, weights, u_fac=1):
    out = np.prod([v ** (w * u_fac) for v, w in zip(vals, weights)], axis=0)
    return out
