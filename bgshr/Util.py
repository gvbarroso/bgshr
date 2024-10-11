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
    df_sub is the dataframe subsetted to a single Na, uR and t

    Cubic spline functions are created over r, for each combination of
    uL and s, returning the fractional reduction based on piR and pi0.

    This function returns the (sorted) arrays of uL and s and the
    dictionary of cubic spline functions with keys (uL, s).
    """
    # Check that only a single entry exists for each item
    all_Ns = np.unique(df_sub["Ns"])
    if not len(all_Ns) == 1:
        all_Ns_split = [[float(_) for _ in Ns.split(";")] for Ns in all_Ns]
        assert len(np.unique([len(_) for _ in all_Ns_split])) == 1
        for vals in zip(np.transpose(all_Ns_split)):
            assert len(np.unique(vals)) == 1

    all_Ts = np.unique(df_sub["Ts"])
    if not len(all_Ts) == 1:
        all_Ts_split = [[float(_) for _ in Ts.split(";")] for Ts in all_Ts]
        assert len(np.unique([len(_) for _ in all_Ts_split])) == 1
        for vals in zip(np.transpose(all_Ts_split)):
            assert len(np.unique(vals)) == 1
 
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


def integrate_with_weights(vals, weights, u_fac=1):
    if len(vals) != len(weights):
        raise ValueError("values and weights are not same length")
    out = np.prod([v ** (w * u_fac) for v, w in zip(vals, weights)], axis=0)
    return out


def _get_time():
    """
    Get a string representing the date and time in mm-dd-yy format.
    """
    return '[' + datetime.strftime(datetime.now(), '%m-%d-%y %H:%M:%S') + ']'


def resolve_elements(element_arrs, L=None):
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


def write_elements(fname, chrom_num, elements, sep='\t', write_header=True):
    """
    Write an array of elements/regions to file in .bed format.
    """
    if not isinstance(chrom_num, str): chrom_num = str(chrom_num)
    open_func = gzip.open if fname.endswith('.gz') else open
    with open_func(fname, 'wb') as file:
        if write_header:
            line = sep.join(['chrom', 'chromStart', 'chromEnd']) + '\n'
            file.write(line.encode())
        for (start, end) in elements:
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
