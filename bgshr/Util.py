"""
Utility functions for `bgshr`, including recombination map handling, reading
input files, and handling input data in lookup tables.
"""

from datetime import datetime
import gzip
import numpy as np
import pandas
from scipy import interpolate
from scipy import integrate
from scipy import stats
import warnings


def load_lookup_table(df_name, sep=","):
    df = pandas.read_csv(df_name, sep=sep)
    return df


def subset_lookup_table(df, generation=0, Ns=None, Ts=None, uL=None):
    if type(Ns) == list:
        raise ValueError("Need to implement")
    if type(Ts) == list:
        raise ValueError("Need to implement")
    
    df_sub = df[df["Generation"] == generation]
    if Ns is not None:
        df_sub = df_sub[df_sub["Ns"] == Ns]
    if Ts is not None:
        df_sub = df_sub[df_sub["Ts"] == Ts]
    if uL is not None:
        df_sub = df_sub[df_sub["uL"] == uL]

    if len(np.unique(df_sub["uL"])) != 1:
        raise ValueError("Require a single uL value - please specify")
    if len(np.unique(df_sub["Ns"])) != 1:
        raise ValueError("Require a single Ns history")
    if len(np.unique(df_sub["Ts"])) != 1:
        raise ValueError("Require a single Ts history")
    return df_sub


def generate_cubic_splines(df_sub):
    """
    df_sub is the dataframe subsetted to a single demography, uR and t

    Cubic spline functions are created over r, for each combination of
    uL and s, returning the fractional reduction based on piR and pi0.

    This function returns the (sorted) arrays of uL and s and the
    dictionary of cubic spline functions with keys (uL, s).
    """

    # Check that only a single entry exists for each item
    assert len(np.unique(np.array(df_sub["Ts"]))) == 1
    assert len(np.unique(np.array(df_sub["Ns"]))) == 1
    assert len(np.unique(df_sub["uR"])) == 1
    assert len(np.unique(df_sub["uL"])) == 1
    assert len(np.unique(df_sub["Generation"])) == 1

    # Get arrays of selection and mutation values
    s_vals = np.array(sorted(list(set(df_sub["s"]))))
    u_vals = np.array(sorted(list(set(df_sub["uL"]))))

    # Store cubic splines of fractional reduction for each pair of u and s, over r
    splines = {}
    for u in u_vals:
        for s in s_vals:
            key = (u, s)
            # Subset to given s and u values
            df_s_u = df_sub[(df_sub["s"] == s) & (df_sub["uL"] == u)]
            rs = np.array(df_s_u["r"])
            Bs = np.array(df_s_u["B"])
            inds = np.argsort(rs)
            rs = rs[inds]
            Bs = Bs[inds]
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


def load_recombination_map(fname, L=None, scaling=1):
    """
    Get positions and rates to build recombination map.

    If L is not None, we extend the map to L if it is greater than the
    last point in the input file, or we truncate the map at L if it is
    less than the last point in the input file.

    If L is not given, the interpolated map does not extend beyond
    the final data point.
    """
    map_df = pandas.read_csv(fname, sep="\\s+")
    pos = np.concatenate(([0], map_df["Position(bp)"]))
    rates = np.concatenate(([0], map_df["Rate(cM/Mb)"])) / 100 / 1e6
    if L is not None:
        if L > pos[-1]:
            pos = np.insert(pos, len(pos), L)
        elif L < pos[-1]:
            cutoff = np.where(L <= pos)[0][0]
            pos = pos[:cutoff]
            pos = np.append(pos, L)
            rates = rates[:cutoff]
        else:
            # L == pos[-1]
            rates = rates[:-1]
    else:
        rates = rates[:-1]
    assert len(rates) == len(pos) - 1
    rmap = build_recombination_map(pos, rates * scaling)
    return rmap


def load_bedgraph_recombination_map(
    fname, 
    sep=",", 
    L=None, 
    scaling=1,
    start_col="start",
    end_col="end",
    rate_col="rate"
):
    """
    Load a recombination map stored in bedgraph format. If L is given, the map
    is truncated to end at position L- if L exceeds the length of the map, an 
    error is raised.
    """
    map_df = pandas.read_csv(fname, sep=sep)
    starts = np.array(map_df[start_col])
    ends = np.array(map_df[end_col])
    rates = np.array(map_df[rate_col])
    edges = np.concatenate(([starts[0]], ends))
    if L is not None:
        if L > ends[-1]:
            raise ValueError("`L` cannot exceed the highest physical position")
        else:
            cutoff = np.where(L <= edges)[0][0]
            edges = np.append(edges[:cutoff], L) 
            rates = rates[:cutoff]
    assert len(rates) == len(edges) - 1
    ratemap = build_recombination_map(edges, rates * scaling)
    return ratemap


def haldane_map_function(rs):
    """
    Returns recombination fraction following Haldane's map function.
    """
    return 0.5 * (1 - np.exp(-2 * rs))


def load_elements(bed_file, L=None):
    """
    From a bed file, load elements. If L is not None, we exlude regions
    greater than L, and any region that overlaps with L is truncated at L.
    """
    elem_left = []
    elem_right = []
    data = pandas.read_csv(bed_file, sep="\t", header=None)
    elem_left = np.array(data[1])
    elem_right = np.array(data[2])
    if L is not None:
        to_del = np.where(elem_left >= L)[0]
        elem_left = np.delete(elem_left, to_del)
        elem_right = np.delete(elem_right, to_del)
        to_trunc = np.where(elem_right > L)[0]
        elem_right[to_trunc] = L
    elements = np.zeros((len(elem_left), 2), dtype=int)
    elements[:, 0] = elem_left
    elements[:, 1] = elem_right
    return elements


def get_elements(df, L=None):
    """
    From a bed file, load elements. If L is not None, we exlude regions
    greater than L, and any region that overlaps with L is truncated at L.
    """
    elem_left = []
    elem_right = []
    df_sub = df[df["selected"] == 1] # select only exons

    elem_left = np.array(df_sub["start"])
    elem_right = np.array(df_sub["end"])
    if L is not None:
        to_del = np.where(elem_left >= L)[0]
        elem_left = np.delete(elem_left, to_del)
        elem_right = np.delete(elem_right, to_del)
        to_trunc = np.where(elem_right > L)[0]
        elem_right[to_trunc] = L
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
    weights[0] += 1 - stats.gamma.cdf(-s_vals[0], shape, scale=scale)
    return weights


def _weights_gamma_dfe(s_vals, shape, scale):
    assert np.all(s_vals <= 0)
    s_vals_sorted = np.sort(s_vals)
    if np.any(s_vals != s_vals_sorted):
        raise ValueError("selection values are not sorted")
    midpoints = (s_vals[1:] + s_vals[:-1]) / 2
    grid = np.concatenate([[-np.inf], midpoints, [0]])
    cdf_evals = stats.gamma.cdf(-grid, shape, scale=scale)
    weights = -np.diff(cdf_evals)
    return weights


def _get_dfe_weights(dfe, s_vals):
    """
    Input DFEs should already be scaled as needed.
    """
    if dfe["type"] == "gamma":
        weights = _weights_gamma_dfe(s_vals, dfe["shape"], dfe["scale"])
    elif dfe["type"] == "gamma_neutral":
        _weights = _weights_gamma_dfe(s_vals, dfe["shape"], dfe["scale"])
        p_neu = dfe["p_neu"]
        weights = np.append(
            _weights[:-1] * (1 - p_neu), _weights[-1] * (1 - p_neu) + p_neu)
    else:
        raise ValueError(f"DFE type {dfe['type']} is unknown")
    return weights


def integrate_with_weights(vals, weights, u_fac=1):
    if len(vals) != len(weights):
        raise ValueError("values and weights are not same length")
    out = np.prod([v ** (w * u_fac) for v, w in zip(vals, weights)], axis=0)
    return out


def convert_bedgraph_mutation_map(
    fname,
    out_fname,
    chrom_col="chr",
    start_col="start", 
    end_col="end", 
    rate_col="u"
):
    """
    Construct a bedgraph file formatted for use in the B prediction pipeline
    from a bedgraph file with only a `rate_col`.
    """
    df = pandas.read_csv(fname)
    starts = np.array(df[start_col])
    ends = np.array(df[end_col])
    avg_mut = np.array(df[rate_col])
    num_sites = ends - starts
    data = {
        "chrom": df[chrom_col],
        "chromStart": starts,
        "chromEnd": ends,
        "num_sites": num_sites, 
        "avg_mut": avg_mut,
        "num_sites_masked": num_sites, 
        "avg_mut_masked": avg_mut
    }
    pandas.DataFrame(data).to_csv(out_fname, index=False)
    return


def load_u_array(mut_tbl_file, masked=True):
    """
    Load mutation rates from a windowed mutation rate table. The following
    columns are expected: chrom, chromStart, chromEnd, num_sites, avg_mut,
    num_sites_masked, avg_mut_masked

    :param mut_tbl_file: Pathname of a .csv/.bedgraph file holding windowed
        mutation rate information.
    :param masked: If True (default), return quantities tabulated following
        the application of a genetic mask. Reads from preexisiting columns
        in the table "num_sites_masked" and "avg_mut_masked".

    :returns: Array of windows, array of windowed site counts, array of 
        windowed mutation rates.
    """
    mut_tbl = pandas.read_csv(mut_tbl_file)
    windows = np.array([mut_tbl["chromStart"], mut_tbl["chromEnd"]]).T
    if masked:
        num_sites = np.array(mut_tbl["num_sites_masked"])
        avg_mut = np.array(mut_tbl["avg_mut_masked"])
    else:
        num_sites = np.array(mut_tbl["num_sites"])
        avg_mut = np.array(mut_tbl["avg_mut"])
    return windows, num_sites, avg_mut


def load_scaled_uL_arrays(
    mut_tbl_file, 
    annot_tbl_files, 
    Ne_scale=1, 
    uL0=1e-8,
    filter_zeros=True
):
    """
    Load arrays of deleterious mutation rate factors for one or more classes of
    functionally constrained elements. These are windowed average rates weighted
    by the number of constrained sites per window, and further scaled by a unit
    mutation rate `uL0` and optionally an effective population size ratio. For 
    use in predicting B values- therefore uses mutation rates tabulated *before* 
    the application of a genetic mask.

    Optionally scales mutation rates. `Ne_scale` is for use with equilibrium
    lookup tables. Call the effective size embodied in a lookup table computed
    for an equilibrium population Ne0. We can predict B with a different Ne
    parameter by scaling u, r and s by the ratio Ne/Ne0. The scaling on u is
    implemented with `Ne_scale`. `uL0` is the deleterious mutation rate modeled
    in the lookup table. 

    :param mut_tbl_file: Pathname of a .csv/.bedgraph file holding windowed
        mutation rate information.
    :param annot_tbl_files: List of pathnames to .csv/.bedgraph files holding
        windowed counts of constrained sites and ratios of deleterious 
        mutation rates to the average rate. Each file corresponds to a class
        of constrained genetic elements.
    :param Ne_scale: Optional linear scale to mutation rates. Accounts for 
        difference in the desired Ne and the Ne (`Ne0`) represented in an 
        equilibrium lookup table (default 1).
    :param u0: Optional mutation rate to scale by (default 1e-8). Should 
        correspond to the mutation rate in the lookup table being used.
        Could be set to 1 to load unscaled rates.
    :param filter_zeros: If True (default), remove all windows where uL is zero  
        in every annotation class from output windows and uL arrays.

    :returns: Array of windows corresponding to uL values, list of uL arrays.
    """
    mut_tbl = pandas.read_csv(mut_tbl_file)
    windows = np.array([mut_tbl["chromStart"], mut_tbl["chromEnd"]]).T
    tot_rates = np.array(mut_tbl["avg_mut"])
    uL_arrs = []
    for file in annot_tbl_files:
        annot_tbl = pandas.read_csv(file)
        _windows = np.array([annot_tbl["chromStart"], annot_tbl["chromEnd"]]).T
        if not np.all(_windows == windows):
            raise ValueError(
                "Annotation/mutation tables have mismatched windows")
        del_sites = np.array(annot_tbl["num_sites"])
        factors = np.array(annot_tbl["scale"])
        unscaled_uL_arr = del_sites * factors * tot_rates
        uL_arr = unscaled_uL_arr * Ne_scale / uL0
        uL_arrs.append(uL_arr)
    if filter_zeros:
        nonzero = np.where(np.sum(uL_arrs, axis=0) > 0)[0]
        uL_windows = windows[nonzero]
        uL_arrs = [uL_arr[nonzero] for uL_arr in uL_arrs]
    return uL_windows, uL_arrs


def load_uL_arrays(mut_tbl_file, annot_tbl_files, masked=True):
    """
    Load arrays recording the average deleterious mutation rate and number of 
    constrained sites for one or more classes of constrained elements.
    
    :param mut_tbl_file: Pathname of a .csv/.bedgraph file holding windowed
        mutation rate information.
    :param annot_tbl_files: List of pathnames to .csv/.bedgraph files holding
        windowed counts of constrained sites and ratios of deleterious 
        mutation rates to the average rate. Each file corresponds to a class
        of constrained genetic elements.
    :param masked: If True (default), return quantities tabulated following
        the application of a genetic mask. Reads from preexisiting columns
        in the table, "num_sites_masked" and "avg_mut_masked".

    :returns: List of arrays of constrained site counts, list of arrays of 
        window-average mutation rates for constrained sites.
    """
    mut_tbl = pandas.read_csv(mut_tbl_file)
    windows = np.array([mut_tbl["chromStart"], mut_tbl["chromEnd"]]).T
    tot_rates = np.array(mut_tbl["avg_mut"])
    uL_arrs = []
    del_sites_arrs = []
    for file in annot_tbl_files:
        annot_tbl = pandas.read_csv(file)
        _windows = np.array([annot_tbl["chromStart"], annot_tbl["chromEnd"]]).T
        if not np.all(_windows == windows):
            raise ValueError(
                "Annotation/mutation tables have mismatched windows")
        if masked:
            del_sites = np.array(annot_tbl["num_sites_masked"])
            factors = np.array(annot_tbl["scale_masked"])
        else:
            del_sites = np.array(annot_tbl["num_sites"])
            factors = np.array(annot_tbl["scale"])
        uL_arr = factors * tot_rates
        uL_arrs.append(uL_arr)
        del_sites_arrs.append(del_sites)
    return del_sites_arrs, uL_arrs


def _get_time():
    """
    Return a string representing the time and date with yy-mm-dd format.
    """
    return '[' + datetime.strftime(datetime.now(), '%y-%m-%d %H:%M:%S') + ']'


def regions_to_mask(regions, L=None):
    """
    Return a boolean mask array that equals 0 within `regions` and 1 
    elsewhere.
    """
    if L is None:
        L = regions[-1, 1]
    mask = np.ones(L, dtype=bool)
    for (start, end) in regions:
        if start > L:
            break
        if end > L:
            end = L
        mask[start:end] = 0
    return mask


def mask_to_regions(mask):
    """
    Return an array representing the regions that are not masked in a boolean
    array (0s).
    """
    jumps = np.diff(np.concatenate(([1], mask, [1])))
    starts = np.where(jumps == -1)[0]
    ends = np.where(jumps == 1)[0]
    regions = np.stack([starts, ends], axis=1)
    return regions


def collapse_regions(regions):
    """
    Collapse any overlapping elements in an array together.
    """
    return mask_to_regions(regions_to_mask(regions))


def intersect_regions(regions_arrs, L=None):
    """
    Form an array of regions from the intersection of sites in input regions
    arrays. These may be mask regions, elements, or whatever.

    :param regions_arrs: List of regions arrays.
    :param L: Maximum position to include (default None).

    :returns: Array of regions composed of shared sites.
    """
    if L is None:
        L = max([regions[-1, 1] for regions in regions_arrs])
    coverage = np.zeros(L, dtype=np.uint8) 
    for elements in regions_arrs:
        for (start, end) in elements:
            coverage[start:end] += 1
    boolmask = coverage < len(regions_arrs)
    isec = mask_to_regions(boolmask)
    return isec


def add_regions(regions_arrs, L=None):
    """
    Form an array of regions from the union of covered sites in several input
    regions arrays.
    """
    if L is None:
        L = max([regions[-1, 1] for regions in regions_arrs])
    coverage = np.zeros(L, dtype=np.uint8) 
    for elements in regions_arrs:
        for (start, end) in elements:
            coverage[start:end] += 1
    boolmask = coverage < 1
    union = mask_to_regions(boolmask)
    return union


def subtract_regions(elements0, elements1, L=None):
    """
    Get an array of regions representing sites that belong to regions in 
    `elements0` and not `elements1`
    """
    if L is None:
        L = max((elements0[-1, 1], elements1[-1, 1]))
    boolmask = np.ones(L, dtype=bool) 
    for (start, end) in elements0:
        boolmask[start:end] = False
    for (start, end) in elements1:
        boolmask[start:end] = True
    ret = mask_to_regions(boolmask)
    return ret


def construct_site_map(intervals, values, L=None):
    """
    Construct a site-resolution map from a set of window starts/ends `intervals`
    and an array of window `values`.
    """
    assert len(intervals) == len(values)
    if L is None:
        L = int(intervals[-1, 1])
    site_map = np.zeros(L, dtype=np.float64)
    for ii, (start, end) in enumerate(intervals):
        if start >= L:
            break
        if end > L:
            end = L
        site_map[start:end] = values[ii]
    return site_map


def compute_windowed_average(intervals, site_map):
    """
    Compute windowed averages of some site-resolution quantity `vec`. Non-
    numeric (nan) values are ignored- windows where all values are nan are
    left with averages of zero. Also returns an array holding the count of
    non-missing site data in each window.

    If intervals exceed the length of site map, no error is raised.

    :param intervals: Array of interval starts/ends.
    :param site_map: Site-resolution ratemap.

    :returns: Arrays of windowed site counts and window averages.
    """
    num_sites = np.zeros(len(intervals), dtype=np.int64)
    avg_rate = np.zeros(len(intervals), dtype=np.float64)
    for ii, (start, end) in enumerate(intervals):
        if np.all(np.isnan(site_map[start:end])):
            continue
        else:
            num_sites[ii] = np.count_nonzero(np.isfinite(site_map[start:end]))
            avg_rate[ii] = np.nanmean(site_map[start:end])
    return num_sites, avg_rate


def read_bedfile(fname, filter_col=None, sep=None):
    """
    Load a bed file, returning an array of intervals and the chromosome number.

    :param dict filter_col: Optional 1-dictionary for filtering intervals. If
        given, return only intervals where the column named by the key of
        `filter_col` has an entry matching `filter_col[key]`.
    :param str sep: Optional separator string, defaults to "\t" if None.
    """
    # Check whether there is a header
    if fname.endswith(".gz"):
        with gzip.open(fname, "rb") as fin:
            first_line = fin.readline().decode()
    else:
        with open(fname, "r") as fin:
            first_line = fin.readline()
    if "," in first_line:
        split_line = first_line.split(",")
        if sep is None:
            sep = ","
    else:
        split_line = first_line.split()
        if sep is None:
            sep = "\s+"
    if split_line[1].isnumeric():
        data = pandas.read_csv(fname, sep=sep, header=None)
    else:
        data = pandas.read_csv(fname, sep=sep)
    if filter_col is not None:
        assert len(filter_col) == 1
        col_name = next(iter(filter_col))
        data = data[data[col_name] == filter_col[col_name]]
    columns = data.columns
    intervals = np.stack(
        (data[columns[1]], data[columns[2]]), axis=1).astype(np.int64)
    uniq_chroms = list(set(data[columns[0]]))
    if len(uniq_chroms) > 1:
        warnings.warn(f"Loaded bed file has more than one unique chrom")
    chrom = str(uniq_chroms[0])
    return intervals, chrom


def write_bedfile(fname, intervals, chrom):
    """
    Write an array of elements/regions to file in .bed format.

    :param str chrom: String representing the chromosome number.
    """
    if not isinstance(chrom, str): 
        chrom = str(chrom)
    data = {
        "chrom": [chrom] * len(intervals), 
        "chromStart": intervals[:, 0],
        "chromEnd": intervals[:, 1]
    }
    pandas.DataFrame(data).to_csv(fname, index=False)
    return


####


def resolve_element_overlaps(element_arrs, L=None):
    """
    Resolve overlaps between classes of elements in a brute-force manner, 
    using order in the list `element_arrs` to determine priority.

    Where overlap exists between arrays, the lower-priority array has its
    overlapping sites excised. 
    """
    if not L:
        L = max(e[-1, 1] for e in element_arrs)
    covered = np.zeros(L)
    resolved_arrs = []
    for elements in element_arrs:
        mask = regions_to_mask(elements, L=L)
        overlaps = np.logical_and(mask == 0, covered == 1)
        print(_get_time(), f'eliminated {overlaps.sum()} overlaps')
        mask[overlaps] = 1
        resolved_arrs.append(mask_to_regions(mask))
        covered[~mask] += 1
    assert not np.any(covered > 1)
    return resolved_arrs


def read_bedgraph(fname, sep=','):
    """
    From a bedgraph-format file, read and return chromosome number(s), an 
    array of genomic regions and a dictionary of data columns. 

    If the file has one unique chromosome number, returns it as a string of
    the form `chr00`; if there are several, returns an array of string
    chromosome numbers of this form for each row.
    Possible file extensions include but are not limited to .bedgraph, .csv,
    and .tsv, with column seperator determined by the `sep` argument.
    """
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, 'rb') as file:
        header_line = file.readline().decode().strip().split(sep)
    # check for proper header format
    assert header_line[0] in ['chrom', '#chrom']
    assert header_line[1] in ['chromStart', 'start']
    assert header_line[2] in ['chromEnd', 'end']
    fields = header_line[3:]
    # handle the return of the chromosome number(s)
    chrom_nums = np.loadtxt(
        fname, usecols=0, dtype=str, skiprows=1, delimiter=sep
    )
    if len(set(chrom_nums)) == 1:
        ret_chrom = chrom_nums[0]
    else:
        # return the whole vector if there are >1 unique chromosome
        ret_chrom = chrom_nums
    windows = np.loadtxt(
        fname, usecols=(1, 2), dtype=int, skiprows=1, delimiter=sep
    )
    cols_to_load = tuple(range(3, len(header_line)))
    arr = np.loadtxt(
        fname,
        usecols=cols_to_load,
        dtype=float,
        skiprows=1,
        unpack=True,
        delimiter=sep
    )
    dataT = [arr] if arr.ndim == 1 else [col for col in arr]
    data = dict(zip(fields, dataT))
    return ret_chrom, windows, data


def write_bedgraph(fname, chrom_num, regions, data, sep=','):
    """
    Write a .bedgraph-format file from an array of regions/windows and a 
    dictionary of data columns.
    """
    for field in data:
        if len(data[field]) != len(regions):
            raise ValueError(f'data field {data} mismatches region length!')
    open_func = gzip.open if fname.endswith('.gz') else open
    fields = list(data.keys())
    header = sep.join(['#chrom', 'chromStart', 'chromEnd'] + fields) + '\n'
    with open_func(fname, 'wb') as file:
        file.write(header.encode())
        for i, (start, end) in enumerate(regions):
            ldata = [str(data[field][i]) for field in fields]
            line = sep.join([chrom_num, str(start), str(end)] + ldata) + '\n'
            file.write(line.encode())
    return
