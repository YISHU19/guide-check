"""M6: how far the comps estimate misses for a TYPED-IN property.

The app judges a guide by its gap to the comps estimate, measured in typical
errors. The yardstick has to be the error of that estimate as the app actually
produces it: a property typed in by suburb, type and land area, whose other
features are borrowed from the suburb's most recent sale. That misses by more
than the model does on a full record (14.3% metro, 16.0% regional), and using
the model's error is the wrong yardstick (see src/selftest.py).

So we measure it directly: take unseen test-set sales, type each one in the way
a user would, and compare the comps estimate with the price it actually sold
for. Median absolute percentage error, by segment and property type.
"""

import json
import warnings

import numpy as np
import pandas as pd

import config as C
import serve

N_PER_CELL = 400


def main():
    warnings.filterwarnings("ignore")
    _, _, _, _, feat, meta, _ = serve.load()
    test = feat.loc[meta["test_ids"]]
    test = test.assign(seg=np.where(test["dist_cbd_km"] < 50, "metro", "regional"))

    out = {}
    for (seg, ptype), g in test.groupby(["seg", "property_type"], observed=True):
        s = g.sample(min(N_PER_CELL, len(g)), random_state=C.SEED)
        ape = []
        for _, r in s.iterrows():
            area = r["land_area_m2"] if pd.notna(r["land_area_m2"]) else None
            try:
                res = serve.value_manual(r["suburb"], r["postcode"], str(ptype),
                                         land_area_m2=area, when=r["contract_date"])
            except ValueError:
                continue
            if res.get("comps_estimate"):
                ape.append(abs(res["comps_estimate"] / r["sale_price"] - 1) * 100)
        out.setdefault(seg, {})[str(ptype)] = round(float(np.median(ape)), 1)
        print(f"    {seg:<8} {ptype:<5} n={len(ape):>4}  median APE {out[seg][str(ptype)]:.1f}%")

    (C.ART / "comps_errors.json").write_text(json.dumps(
        {"median_ape_pct": out, "n_per_cell": N_PER_CELL, "seed": C.SEED,
         "method": "test-set sales typed in via serve.value_manual"}, indent=2))
    print(f"    saved {C.ART / 'comps_errors.json'}")


if __name__ == "__main__":
    print("M6  measuring typed-in comps error ...")
    main()
