from glob import glob
import os
import pickle
import re

import numpy as np


def read_density_matrices(
    n,
    save_dir=f"./data/normalized_gibbs/",
    read_up_as_dict=False,
    special=False,
):
    recs = []
    pattern = (
        r"n(\d+)_nt(\d+)_s(\d+).pkl" if not special else r"n(\d+)_type(zero|ghz)_s0.pkl"
    )

    glob_pattern = f"n{n}_nt*_s*.pkl" if not special else f"n{n}_type*_s0.pkl"

    for path in glob(os.path.join(save_dir, glob_pattern)):
        m = re.search(pattern, path)
        if m:
            with open(path, "rb") as f:
                rho, terms, up = pickle.load(f)

            if read_up_as_dict:
                labels = sorted(up.keys())
                up = np.array([up[l] for l in labels])

            rec = {
                "up": up,
                "rho": rho,
                "terms": terms,
            }

            if special:
                n_, t_ = m.groups()
                rec["n"] = int(n_)
                rec["nt"] = 0
                rec["type"] = t_
                rec["seed"] = 0
            else:
                n_, nt, s = m.groups()
                rec["n"] = int(n_)
                rec["nt"] = int(nt)
                rec["type"] = "gibbs"
                rec["seed"] = int(s)

            recs.append(rec)
    return recs
