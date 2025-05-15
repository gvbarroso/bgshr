import numpy as np
import warnings
from scipy import interpolate
import copy
import pandas
from multiprocessing import Pool

from . import Util, ClassicBGS


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
    """
    Compute a matrix of recombination distance between focal neutral sites `xs`
    and constrained elements/windows `elements`. 
    """
    # element midpoints
    x_midpoints = np.mean(elements, axis=1)
    # get recombination map values
    r_midpoints = rmap(x_midpoints)
    r_xs = rmap(xs)
    r_dists = np.zeros((len(elements), len(xs)))
    # get recombination fractions
    r_dists[:, :] = Util.haldane_map_function(
        np.abs(r_xs[None, :] - r_midpoints[:, None])
    )
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


def load_extended_lookup_table(file):
    """
    Load a lookup table and extend it using the ClassicBGS module. 

    :param file: Pathname of lookup table file (.csv). 

    :returns: Extended lookup table, array of selection coefficients, dictionary
        of cubic splines, and table Ne. When the table has several size epochs, 
        table Ne is returne as None.
    """
    df = Util.load_lookup_table(file)
    df_sub = Util.subset_lookup_table(df)
    r_vals = np.unique(df_sub["r"])
    if np.max(r_vals) < 0.5:
        raise ValueError("maximum r value in table must equal 0.5")
    s_min = np.min(df_sub['s'])
    s_extend = -np.concatenate(([0], np.logspace(-6, 0, 46)))
    s_extend = s_extend[s_extend < s_min]
    Ts = next(iter(df_sub["Ts"]))
    Ns = next(iter(df_sub["Ns"]))
    # Equilibrium tables
    if len(str(Ts).split(";")) == 1:
        s_min = np.min(df_sub['s'])
        s_extend = -np.concatenate(([0], np.logspace(-6, 0, 46)))
        s_extend = s_extend[s_extend < s_min]
        df_extended = ClassicBGS.extend_lookup_table(df_sub, s_extend)
        Ne0 = next(iter(df_extended['Ns']))
    # Non-equilibrium tables
    else:
        Ts_split = [float(T) for T in Ts.split(";")]
        Ns_split = [int(N) for N in Ns.split(";")]
        df_extend = ClassicBGS.build_lookup_table_n_epoch(
            s_extend, r_vals, Ns_split, Ts_split
        )
        df_extended = pandas.concat((df_sub, df_extend), ignore_index=True)
        df_extended["Ts"] = Ts
        df_extended["Ns"] = Ns
        Ne0 = None
    u_vals, s_vals, splines = Util.generate_cubic_splines(df_extended)
    return df_extended, s_vals, splines, Ne0


def _expected_del_pi0(dfes, s_vals, del_sites_arrs, u_arrs, Ne=None, u0=1e-8):
    """

    For use with tables with equilibrium demographic histories.
    """
    Hls = 2 * np.array([ClassicBGS._get_Hl(s, Ne, u0) for s in s_vals])
    sum_del_pi0 = np.zeros(len(u_arrs[0]), dtype=np.float64)
    for (dfe, u_arr, del_sites) in zip(dfes, u_arrs, del_sites_arrs):
        weights = Util._get_dfe_weights(dfe, s_vals)
        unit_del_pi0 = np.sum(weights * Hls)
        sum_del_pi0 += unit_del_pi0 * u_arr * del_sites
    del_pi0 = np.zeros(len(del_sites), dtype=np.float64)
    tot_del_sites = np.sum(del_sites_arrs, axis=0)
    del_pi0[tot_del_sites > 0] = (
        sum_del_pi0[tot_del_sites > 0] / tot_del_sites[tot_del_sites > 0])
    return del_pi0


def _tbl_expected_del_pi0(df, dfes, del_sites_arrs, u_arrs):
    """
    Compute window-average pi0 for constrained sites under direct selection from
    values stored in a lookup table. For use with tables with non-equilibrium
    demographic histories.
    """
    df_sub = df[df["r"] == 0]
    u0 = np.unique(df_sub["uL"])[0]
    s_vals = np.sort(np.unique(df_sub["s"]))
    Hls = 2 * np.array([df_sub[df_sub["s"] == s]["Hl"] for s in s_vals])
    sum_del_pi0 = np.zeros(len(u_arrs[0]), dtype=np.float64)
    for (dfe, u_arr, del_sites) in zip(dfes, u_arrs, del_sites_arrs):
        weights = Util._get_dfe_weights(dfe)
        unit_del_pi0 = np.sum(Hls * weights)
        sum_del_pi0 += unit_del_pi0 * u_arr * del_sites
    del_pi0 = np.zeros(len(del_sites), dtype=np.float64)
    tot_del_sites = np.sum(del_sites_arrs, axis=0)
    del_pi0[tot_del_sites > 0] = (
        sum_del_pi0[tot_del_sites > 0] / tot_del_sites[tot_del_sites > 0])
    return del_pi0


def parallel_Bvals(
    xs,
    s_vals,
    splines,
    dfes,
    uL_arrs,
    uL_windows,
    rmap=None,
    r=None,
    B_map=None,
    B_elem=None,
    tolerance=None,
    df=None,
    block_size=5000,
    cores=None
):
    """
    Predict a B-landscape at focal neutral sites along a chromosome for one or 
    more classes of functionally constrained elements.

    :param xs: Array holding the positions of focal neutral sites.
    :param s_vals: Array of selection coefficients. These should be a subset of
        the s-values in `splines`.
    :param splines: A dictionary that maps (u, s) tuples to cubic splines which
        interpolate "unit" B values as a function of recombination distance. 
    :param dfes: A list of dictionaries defining parameters of gamma-distributed 
        DFEs. These must have keys "shape", "scale" and "type". "type" should 
        map to "gamma" or "gamma_neutral"; if "gamma_neutral", an additional
        parameter "p_neu" defining the fraction of neutral mutations is needed.
    :param uL_arrs: List of arrays holding deleterious mutation factors. Each 
        element in an array corresponds to a window in `uL_windows`. Scaling is 
        twofold: first, uL is a product of the average deleterious mutation rate 
        and the number of constrained sites that belong to the class of interest 
        in each window. Second, uL is normalized by u0, the mutation rate that 
        was used to build the input lookup table- and under equilibrium 
        demographies where the Ne for which we wish to predict differs from the 
        Ne used to build the lookup table, uL is also scaled by Ne / Ne0 where 
        Ne0 is the table Ne.
    :param uL_windows: Array recording windows that hold constrained sites. The
        density of constrained sites in each window is reflected in `uL_arrs`.
    :param rmap: Recombination map function (default None builds a uniform map).
    :param r: Optional uniform recombination map rate (defaults to 1e-8).
    :param B_map: Optional function, interpolating physical coordinates to B
        values from a prior prediction- used to rescale local parameters in the
        interference correction. Mutually exclusive with `B_elem` (default None)
    :param B_elem: Optional array specifying average B values in each 
        constrained window, obtained from a prior round of prediction. For use
        in the interference correction and mutually exclusive with `B_map`
        (default None)
    :param tolerance:
    :param df: Lookup table, loaded as a pandas dataframe. Required when 
        specifying a `tolerance` (default None).
    :param block_size: Number of focal neutral sites/constrained elements to 
        handle at once on each core (default 5000). Expect that memory 
        requirements scale as the square of this quantity at least.
    :param cores: Number of cores across which to parallelize predictions. If
        1 or None, simply loops over pairs of neutral/constrained blocks.
    
    :returns: An array of predicted B-values matching `xs` in shape.
    """
    # If a B-map was given, prepare to perform interference correction
    if B_map is not None:
        if B_elem is None:
            B_elem = _get_B_per_element(B_map, uL_windows)
        else:
            raise ValueError("You cannot give both `B_elem` and `B_map`")

    # Build a uniform recombination map if one was not provided
    if rmap is None:
        L = max([xs[-1], uL_windows[-1][1]])
        if r is None:
            print("No recombination rates provided, assuming r=1e-8")
            r = 1e-8
        rmap = Util.build_uniform_rmap(r, L)

    if np.isscalar(s_vals):
        s_vals = np.array([s_vals])

    # If an error tolerance is provided, find a maximum r value for each s
    if tolerance is not None:
        if df is None:
            raise ValueError(
                "You must provide a lookup table to use `tolerance`")
        thresholds = _get_r_thresholds(df, max_err=tolerance)
    else:
        thresholds = None

    # Group focal sites and constrained elements into windows
    build_blocks = lambda xs, size: [
        np.arange(start_idx, end_idx) for start_idx, end_idx in 
        zip(np.arange(0, len(xs), size), 
            np.append(np.arange(size, len(xs), size), len(xs)))
    ]
    xblocks = build_blocks(xs, block_size)
    ublocks = build_blocks(uL_windows, block_size)

    Bs = np.ones(len(xs), dtype=np.float64)

    if cores is None or cores == 1:
        for xblock in xblocks:
            for ublock in ublocks:
                block_uL_arrs = [uLs[ublock] for uLs in uL_arrs]
                if B_elem is not None:
                    block_B_elem = B_elem[ublock]
                else:
                    block_B_elem = None
                Bs[xblock] *= _predict_block_B(
                    xs[xblock],
                    dfes,
                    uL_windows[ublock],
                    block_uL_arrs,
                    splines,
                    rmap,
                    thresholds=thresholds,
                    B_elem=block_B_elem
                )       
    else:
        pairings = [(xb, ub) for xb in xblocks for ub in ublocks]
        groups = [pairings[i:i + cores] for i in range(0, len(pairings), cores)]
        for group in groups:
            args = []
            for xblock, ublock in group:
                block_uL_arrs = [uLs[ublock] for uLs in uL_arrs]
                if B_elem is not None:
                    block_B_elem = B_elem[ublock]
                else:
                    block_B_elem = None
                args.append((
                    xs[xblock],
                    dfes,
                    uL_windows[ublock],
                    block_uL_arrs,
                    splines,
                    rmap,
                    thresholds,
                    block_B_elem
                ))
            with Pool(len(group)) as p:
                group_Bs = p.starmap(_predict_block_B, args)
            for (xblock, _), window_B in zip(group, group_Bs):
                Bs[xblock] *= window_B
    return Bs


def _predict_block_B(    
    xs,
    dfes,
    s_vals,
    splines,
    uL_arrs,
    uL_windows,
    rmap,
    thresholds=None,
    B_elem=None
):
    """
    
    """
    pairwise_B = _predict_block_B_pairwise(
        xs,
        s_vals,
        splines,
        uL_arrs,
        uL_windows,
        rmap,
        thresholds=thresholds,
        B_elem=B_elem
    )
    B_vec = np.ones(len(xs))
    for B_vals, dfe in zip(pairwise_B, dfes):
        weights = Util._get_dfe_weights(dfe)
        B_vec *= Util.integrate_with_weights(B_vals, weights[:-1])
    return B_vec


def _predict_block_B_pairwise(
    xs,
    s_vals,
    splines,
    uL_arrs,
    uL_windows,
    rmap,
    thresholds=None,
    B_elem=None
):
    """
    
    """
    u0 = np.unique([k[0] for k in splines.keys()])[0]
    r_dists = _get_r_dists(xs, uL_windows, rmap)
    pairwise_B = [np.ones((len(s_vals[:-1]), len(xs))) for uLs in uL_arrs]
    # Initial predictions
    if B_elem is None:
        for ii, s_val in enumerate(s_vals[:-1]):
            if thresholds is not None:
                if np.all(r_dists > thresholds[ii]):
                    continue
            unit_B = splines[(u0, s_val)](r_dists)
            for jj, uL_arr in enumerate(uL_arrs):
                pairwise_B[jj][ii] = np.prod(unit_B ** uL_arr[:, None], axis=0)
    # Predictions under the interference correction
    else:
        for ii, s_val in enumerate(s_vals[:-1]):
            if thresholds is not None:
                if np.all(r_dists > thresholds[ii]):
                    continue
            s_elems = s_val * B_elem
            unit_B = _get_interpolated_B_vals(
                xs, s_elems, s_vals, r_dists, splines)
            for jj, uLs in enumerate(uL_arrs):
                pairwise_B[jj][ii] = np.prod(unit_B ** uLs[:, None], axis=0)
    return pairwise_B


def _adjust_uL_arrays(B_map, uL_arrs, uL_windows):
    """
    Scale the deleterious mutation rate factor uL by local B-value. Used in the
    interference correction. 

    :param B_map: 
    :param uL_arrs:
    :param uL_windows: 
    """
    window_B = _get_B_per_element(B_map, uL_windows)
    adjusted_uL_arrays = [uLs * window_B for uLs in uL_arrs]
    return adjusted_uL_arrays


def _get_interpolated_B_vals(xs, s_elems, s_vals, r_dists, splines, u0=1e-8):
    """
    Interpolate B values 
    """
    unit_B = np.zeros((len(s_elems), len(xs)))
    for i, s_elem in enumerate(s_elems):
        s0, s1, p0, p1 = _get_interpolated_svals(s_elem, s_vals)
        fac0 = p0 * splines[(u0, s0)](r_dists[i])
        fac1 = p1 * splines[(u0, s1)](r_dists[i])
        unit_B[i] = fac0 + fac1
    return unit_B


def _get_r_thresholds(df, tolerance=1e-10):
    """
    
    :param df: Lookup table loaded as a pandas dataframe.
    :param tolerance: Maximum allowable deviation in B-value from 1 
        (default 1e-10).
    """
    s_vals = np.sort(np.unique(df["s"]))
    thresholds = np.zeros(len(s_vals), dtype=np.float64)
    for i, s in enumerate(s_vals):
        B_vals = np.array(df[df["s"] == s]["B"])
        r_vals = np.array(df[df["s"] == s]["r"])
        assert np.all(np.sort(r_vals) == r_vals)
        deviations = 1 - B_vals
        beyond_tolerance = np.where(deviations < tolerance)[0]
        # If all deviations are beyond tolerance, make the threshold r=0.5
        if len(beyond_tolerance) == 0:
            thresholds[i] = 0.5
        else:
            threshold_idx = beyond_tolerance[0]
            thresholds[i] = r_vals[threshold_idx]
    return thresholds
