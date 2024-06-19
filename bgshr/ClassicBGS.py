import numpy as np
import pandas
from scipy import linalg

from . import Util


def reduction_CBGS(s, u, r, L=1):
    """
    This is the result given by Nordborg using either a diffusion (1996, wich
    Charlesworth) or a Markov chain (1996) approach.

    :param s: The negative selection coefficient.
    :param u: The deleterious mutation rate.
    :param r: The recombination rate between the focal site and the selected
        locus.
    :param L: Optionally, a scaling factor for the mutation rate, assuming a
        non-recombining selected locus of length L, and per-base mutation rate
        of `u`.
    """
    return np.exp(-u * L / (-s * (1 + r * (1 + s) / -s) ** 2))


def classic_BGS(xs, s, u, L=None, rmap=None, elements=[]):
    """
    Compute classic BGS reduction due to selection in constrained elements.

    B values are computed from the midpoints of elements. If elements are
    large, consider splitting into smaller regions using `Util.xzy()`.

    THis assumes that we are at steady state.

    :param xs: An arary of positions to compute B-values at.
    :param s: The selection coefficient for each deleterious mutation.
    :param u: Optional, the length of the sequence. If it is not given, we set
        it to the largest value found in `xs` or the end of the final element,
        whichever is larger.
    :rmap: An interpolated recombination map funcion. If a recombination map
        is not provided, we assume the region is non-recombining.
    :param elements: A (sorted) list of non-overlapping [l, r] regions defining
        selected elements.
    """
    B = np.ones(len(xs))
    if len(elements) == 0:
        return B

    if L is None:
        L = max([xs[-1], elements[-1][1]])

    if rmap is None:
        r = 0
        rmap = Util.build_uniform_rmap(r, L)

    r_xs = rmap(xs)
    for e in elements:
        mid = np.mean(e)
        L_elem = e[1] - e[0]
        r_mid = rmap(mid)
        r_dists = np.abs(r_xs - r_mid)
        B *= reduction_CBGS(s, u, r_dists, L=L_elem)
    return B


def extend_lookup_table(df_sub, ss):
    """
    Extend a lookup table at present recombination values for given s values.
    """
    r_vals = np.array(sorted(list(set(df_sub["r"]))))
    cols = df_sub.columns
    data = {
        "Na": np.unique(df_sub["Na"])[0],
        "N1": np.unique(df_sub["Na"])[0],
        "t": np.unique(df_sub["t"])[0],
        "uL": np.unique(df_sub["uL"])[0],
        "uR": np.unique(df_sub["uR"])[0],
        "Order": 0,
        "Generation": 0,
    }
    data["pi0"] = 2 * data["Na"] * data["uR"]

    new_data = []
    for s in ss:
        Bs = reduction_CBGS(s, data["uL"], r_vals)
        data["s"] = s
        data["Hl"] = _get_Hl(s, data["Na"], data["uL"])
        Hrs = Bs * data["pi0"]
        data["piN_pi0"] = data["Hl"] / data["pi0"]
        for r, B, Hr in zip(r_vals, Bs, Hrs):
            data["r"] = r
            data["B"] = B
            data["Hr"] = Hr
            data["piN_piS"] = data["Hl"] / data["Hr"]
            new_row = [data[k] for k in df_sub.columns]
            new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=df_sub.columns)
    df_comb = pandas.concat((df_sub, df_new), ignore_index=True)
    return df_comb


def build_lookup_table(ss, rs, Ne=1e4, uL=1e-8, uR=1e-8):
    """
    Given a list of selection coefficients and rcombination rates, build a
    a diversity-reduction lookup table using just classic background
    selected theory.
    """
    cols = [
        "Na",
        "N1",
        "t",
        "r",
        "s",
        "uL",
        "Order",
        "Generation",
        "Hr",
        "pi0",
        "B",
        "uR",
        "Hl",
        "piN_pi0",
        "piN_piS",
    ]
    data = {
        "Na": Ne,
        "N1": Ne,
        "t": 0,
        "uL": uL,
        "uR": uR,
        "Order": 0,
        "Generation": 0,
    }
    data["pi0"] = 2 * data["Na"] * data["uR"]
    new_data = []
    for s in ss:
        if s == 0:
            Bs = np.ones(len(rs))
        else:
            Bs = reduction_CBGS(s, uL, rs)
        data["s"] = s
        data["Hl"] = _get_Hl(s, Ne, uL)
        data["piN_pi0"] = data["Hl"] / data["pi0"]
        Hrs = Bs * data["pi0"]
        for r, B, Hr in zip(rs, Bs, Hrs):
            data["r"] = r
            data["B"] = B
            data["Hr"] = Hr
            data["piN_piS"] = data["Hl"] / data["Hr"]
            new_row = [data[k] for k in cols]
            new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=cols)
    return df_new


def _get_Hl(s, Ne, u):
    """
    From integrating Eq. 31 in Evans et al (2007) against `4*Ne*u*x*(1-x)`.
    """
    if s == 0:
        return 2 * Ne * u
    else:
        return 4 * Ne * u * np.exp(4 * Ne * s) / (np.exp(4 * Ne * s) - 1) - u / s


###################################################################
# Functions to compute classic BGS predictions with size changes. #
###################################################################


def _sub_intensity_matrix(N_epoch, u, r, s, Ne):
    """
    Given in Nordborg (1997), this is the sub-intensity matrix for the
    structured coalescence model with background selection. It has been
    adjusted to allow for different population size in a given epoch
    from the Ne used to scale time.
    """
    q = u / -s
    p = 1 - q
    b12 = q * r
    b21 = p * (-s + r)
    nu = N_epoch / Ne  # size relative to Ne
    S = np.array(
        [
            [-4 * Ne * b12 - 1 / (nu * p), 4 * Ne * b12, 0],
            [2 * Ne * b21, -2 * Ne * b12 - 2 * Ne * b21, 2 * Ne * b12],
            [0, 4 * Ne * b21, -4 * Ne * b21 - 1 / (nu * q)],
        ]
    )
    return S


def _probability_absorption(N_epoch, u, r, s, Ne, gens):
    """
    Time is measured in units of 2Ne generations. This returns the probability
    of coalescence in the structured coalescent model for BGS in an epoch of a
    given size.
    """
    t = gens / 2 / Ne
    S = _sub_intensity_matrix(N_epoch, u, r, s, Ne)
    q = u / -s
    p = 1 - q
    alpha = np.array([p ** 2, 2 * p * q, q ** 2])
    return 1 - alpha.dot(linalg.expm(S * t)).dot([1, 1, 1])


def _bgs_coalescent_rate(u, r, s, nu):
    """
    The parameter lambda in Nordborg (1997). The parameter `nu` is the relative
    size of the epoch compared to a reference Ne.
    """
    q = u / -s
    p = 1 - q
    b12 = q * r
    b21 = p * (-s + r)
    return b21 ** 2 / (b12 + b21) ** 2 / p / nu + b12 ** 2 / (b12 + b21) ** 2 / q / nu


def expected_tmrca_n_epoch_neutral(Ns, Ts):
    """
    Ns and Ts are vectors of the same length, specifying (piecewise constant)
    population sizes and epoch break points. `Ns` and `Ts` must have the same
    length, and time is measured in generations into the past. The first entry
    in `Ts` should be zero, and values should be monotonically increasing and
    not include infinity.  We assume the last entry in `Ns` specifies the
    steady-state population size prior to any size changes in the past.
    """
    p_coal = 0
    ET = 0
    for N, T0, T1 in zip(Ns, Ts[:-1], Ts[1:]):
        gens = T1 - T0
        # probability of coalescence within this epoch
        p_coal_epoch = 1 - np.exp(-gens / 2 / N)
        # weighted by the probability that we haven't coalescece before this epoch
        epoch_weight = p_coal_epoch * (1 - p_coal)
        p_coal += epoch_weight
        # expected TMRCA conditional on coalescing within this epoch
        ET_cond = 2 * N - gens * (1 - p_coal_epoch) / p_coal_epoch
        # add to ET
        ET += epoch_weight * (T0 + ET_cond)
    # final epoch extending to infinity
    assert p_coal < 1
    ET += (1 - p_coal) * (Ts[-1] + 2 * Ns[-1])
    return ET


def expected_tmrca_n_epoch_bgs(Ns, Ts, u, r, s):
    """
    Given a piecewise-constant population size history, get the expected TMRCA,
    in units of Ne generations. Ne is taken to be the size from the most
    ancient epoch. The expected TMRCA under classic BGS is an extension of the
    result from Nordborg (1997), conditioning on coalescence occurring within a
    given epoch, and then weighting total contribution to TMRCA across
    probability of coalescence occurring within each epoch.

    Ns and Ts must be the same length, and Ts must start at zero and be
    monotonically increasing. Then the size between Ts[0] and Ts[1] is
    Ns[0], the size between Ts[1] and Ts[2] is Ns[1], and so on. The final
    epoch estends from Ts[-1] to infinity, and has size Ns[-1] (== Ne).
    """
    if len(Ns) != len(Ts):
        raise ValueError("Ns and Ts must be the same length")
    p_coal = 0
    ET = 0
    Ne = Ns[-1]
    for N, T0, T1 in zip(Ns, Ts[:-1], Ts[1:]):
        gens = T1 - T0
        # probability of coalescence within this epoch
        p_coal_epoch = _probability_absorption(N, u, r, s, Ne, gens)
        # weighted by the probability that we haven't coalescece before this epoch
        epoch_weight = p_coal_epoch * (1 - p_coal)
        p_coal += epoch_weight
        # expected TMRCA conditional on coalescing within this epoch
        ET_cond = (
            2 * Ne / _bgs_coalescent_rate(u, r, s, N / Ne)
            - gens * (1 - p_coal_epoch) / p_coal_epoch
        )
        # add to ET
        ET += epoch_weight * (T0 + ET_cond)
    # final epoch extending to infinity
    assert p_coal < 1
    ET += (1 - p_coal) * (Ts[-1] + 2 * Ne / _bgs_coalescent_rate(u, r, s, 1))
    return ET


def reduction_CBGS_n_epoch(Ns, Ts, s, u, r, L=1):
    """
    Expected B-value (diversity reduction) for a size change history under
    classical BGS theory.

    Ns and Ts must be the same length, and Ts must start at zero and be
    monotonically increasing. Then the size between Ts[0] and Ts[1] is
    Ns[0], the size between Ts[1] and Ts[2] is Ns[1], and so on. The final
    epoch estends from Ts[-1] to infinity, and has size Ns[-1] (== Ne).
    """
    TBGS = expected_tmrca_n_epoch_bgs(Ns, Ts, u, r, s)
    Tneu = expected_tmrca_n_epoch_neutral(Ns, Ts)
    return (TBGS / Tneu) ** L


def _shift_Ns_Ts(Ns, Ts, gen):
    ## TODO: this is not elegant, and not tested
    coal_gen = Ts[-1] - gen
    if coal_gen >= Ts[-1]:
        return [Ns[-1]], [0]
    # shift time as needed
    Ns_gen = []
    Ts_gen = []
    for i, (N, T0, T1) in enumerate(zip(Ns[:-1], Ts[:-1], Ts[1:])):
        if T0 >= coal_gen:
            Ns_gen.append(N)
            Ts_gen.append(T0 - coal_gen)
        elif T1 > coal_gen:
            Ns_gen.append(N)
            Ts_gen.append(0)
    Ns_gen.append(Ns[-1])
    Ts_gen.append(Ts[-1] - coal_gen)
    return Ns_gen, Ts_gen


def build_lookup_table_n_epoch(ss, rs, Ns, Ts, generations=None, uL=1e-8, uR=1e-8):
    cols = [
        "Na",
        "N1",
        "t",
        "r",
        "s",
        "uL",
        "Order",
        "Generation",
        "Hr",
        "pi0",
        "B",
        "uR",
        "Hl",
        "piN_pi0",
        "piN_piS",
    ]
    Ne = Ns[-1]
    if generations is None:
        generations = [Ts[-1]]

    data = {
        "Na": Ne,
        "N1": Ne,
        "t": 0,
        "uL": uL,
        "uR": uR,
        "Order": 0,
    }
    new_data = []
    for gen in generations:
        data["Generation"] = gen
        Ns_gen, Ts_gen = _shift_Ns_Ts(Ns, Ts, gen)
        # fill in data
        data["pi0"] = expected_tmrca_n_epoch_neutral(Ns_gen, Ts_gen) * uL
        for s in ss:
            data["s"] = s
            data["Hl"] = np.nan  ## TODO
            data["piN_pi0"] = data["Hl"] / data["pi0"]
            for r in rs:
                if s == 0:
                    data["B"] = 1
                else:
                    data["B"] = reduction_CBGS_n_epoch(Ns_gen, Ts_gen, s, uL, r)
                data["Hr"] = data["B"] * data["pi0"]
                data["r"] = r
                data["piN_piS"] = data["Hl"] / data["Hr"]
                new_row = [data[k] for k in cols]
                new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=cols)
    return df_new
