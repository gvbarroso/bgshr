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


def load_bedgraph(fname, sep=",", L=None, scaling=1):
    """
    Get positions and rates to build rate map.
    If L is not None, we extend the map to L if it is greater than the
    last point in the input file, or we truncate the map at L if it is
    less than the last point in the input file.
    If L is not given, the interpolated map does not extend beyond
    the final data point.
    """
    map_df = pandas.read_csv(fname, sep=sep)
    ends = np.concatenate(([0], map_df["end"]))
    rates = np.concatenate(([0], map_df[map_df.columns[3]]))
    if L is not None:
        if L > ends[-1]:
            ends = np.insert(ends, len(ends), L)
        elif L < ends[-1]:
            cutoff = np.where(L <= ends)[0][0]
            ends = ends[:cutoff]
            ends = np.append(ends, L)
            rates = rates[:cutoff]
        else:
            rates = rates[:-1]
    else:
        rates = rates[:-1]
    assert len(rates) == len(ends) - 1
    ratemap = build_recombination_map(ends, rates * scaling)
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
        u_arr = np.array(mut_tbl["avg_mut_masked"])
    else:
        num_sites = np.array(mut_tbl["num_sites"])
        u_arr = np.array(mut_tbl["avg_mut"])
    return windows, num_sites, u_arr


def load_del_uL_arrays(
    mut_tbl_file, 
    annot_tbl_files, 
    Ne_scale=1, 
    u0=1e-8,
    shorten=True
):
    """
    Load arrays recording del_mut * del_sites for one or more classes of 
    functionally constrained elements. For use in predicting B values- therefore
    uses mutation rates tabulated *before* the application of a genetic mask.

    Incorporates two forms of scaling on the mutation rate: (1) scaling to a
    unit mutation rate (often 1e-8) obtained from the lookup table, and (2) 
    scaling to some Ne (`Ne0`, often 1e4) corresponding to the Ne recorded in
    the lookup table. The second scaling should only be imposed when the 
    lookup table has been computed for an equilibrium population.

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
    :param shorten: If True (default), remove all windows where uL is zero  
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
        uL_arr = unscaled_uL_arr * Ne_scale / u0
        uL_arrs.append(uL_arr)
    if shorten:
        nonzero = np.where(np.sum(uL_arrs, axis=0) > 0)[0]
        uL_windows = windows[nonzero]
        uL_arrs = [uL_arr[nonzero] for uL_arr in uL_arrs]
    return uL_windows, uL_arrs


def load_del_u_arrays(mut_tbl_file, annot_tbl_files, masked=True):
    """
    Load arrays recording the deleterious mutation rate and number of 
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
    u_arrs = []
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
        u_arr = factors * tot_rates
        u_arrs.append(u_arr)
        del_sites_arrs.append(del_sites)
    return del_sites_arrs, u_arrs


def _get_time():
    """
    Return a string representing the time and date with yy-mm-dd format.
    """
    return '[' + datetime.strftime(datetime.now(), '%y-%m-%d %H:%M:%S') + ']'


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


def filter_elements_by_length(elements, min_length):
    """
    Remove any intervals with length < min_len from an array of elements.
    """
    above_thresh = np.where(np.diff(elements, axis=1) >= min_length)[0]
    return elements[above_thresh]


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


def _collapse_elements(elements):
    """
    Collapse any overlapping elements in an array together.
    """
    return mask_to_regions(regions_to_mask(elements))


def write_bedfile(fname, chrom_num, regions, sep='\t', write_header=False):
    """
    Write an array of elements/regions to file in .bed format.
    """
    if not isinstance(chrom_num, str): chrom_num = str(chrom_num)
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, 'wb') as file:
        if write_header:
            line = sep.join(['#chrom', 'chromStart', 'chromEnd']) + '\n'
            file.write(line.encode())
        for (start, end) in regions:
            line = sep.join([chrom_num, str(start), str(end)]) + '\n'
            file.write(line.encode())
    return


def setup_windows(size, L, start=0):
    """
    Get an array of contiguous windows of uniform size `size`.
    """
    edges = np.arange(start + size, L + size, size)
    _edges = np.concatenate(([start], edges[:-1].repeat(2), [edges[-1]]))
    return _edges.reshape(-1, 2)


def read_bedfile(fname):
    """
    This is redundant with functions above, but it also returns the
    chromosome number which is sometimes useful. 
    """
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, 'rb') as file:
        line0 = file.readline().decode().split()
    if line0[1].isnumeric() and line0[2].isnumeric():
        skiprows = 0
    else:
        skiprows = 1
    regions = np.loadtxt(
        fname, dtype=np.int64, skiprows=skiprows, usecols=(1, 2)
    )
    if regions.ndim == 1:
        regions = regions[np.newaxis]
    chrom_nums = np.loadtxt(
        fname, dtype=str, skiprows=skiprows, usecols=0
    )
    if len(set(chrom_nums)) == 1:
        ret_chrom = chrom_nums[-1]
    else:
        ret_chrom = chrom_nums
    return ret_chrom, regions
