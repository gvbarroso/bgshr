import numpy as np
import pandas

from . import Util


def reduction_CBGS(s, u, r, L=1):
    """
    TODO: Is this the Nordborg result?
    """
    return np.exp(-u * L / (-s * (1 + r * (1 + s) / -s) ** 2))


def classic_BGS(xs, s, u, L=None, rmap=None, elements=[]):
    """
    Compute classic BGS reduction due to selection in constrained elements.

    B values are computed from the midpoints of elements. If elements are large,
    consider splitting into smaller regions using `Util.xzy()`.

    :param xs:
    :param elements: A (sorted) list of non-overlapping [l, r] regions
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
        data["Hl"] = 0  ## TODO
        Hrs = Bs * data["pi0"]
        for r, B, Hr in zip(r_vals, Bs, Hrs):
            data["r"] = r
            data["B"] = B
            data["Hr"] = Hr
            data["piN_pi0"] = data["Hl"] / data["pi0"]
            data["piN_piS"] = data["Hl"] / data["Hr"]
            new_row = [data[k] for k in df_sub.columns]
            new_data.append(new_row)
    df_new = pandas.DataFrame(new_data, columns=df_sub.columns)
    df_comb = pandas.concat((df_sub, df_new), ignore_index=True)
    return df_comb
