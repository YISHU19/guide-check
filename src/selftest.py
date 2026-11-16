"""Does the guide assessment detect a known gap? 400 labelled cases.

100 unseen test-set sales, each typed in the way a user would and checked with
a guide at 100%, 90%, 80% and 70% of the price it actually sold for. A guide at
the sale price is honest; one at 70-80% is a real 20-30% underquote.

A case counts as flagged when severity is "strong" or "notable". We report the
catch rate on the 70-80% guides and the false-alarm rate on the 100% guides,
under two yardsticks for the typical error:

  model  -- the model's own error on full records (what the bands first used)
  comps  -- the measured error of the comps estimate for a typed-in property
            (artefacts/comps_errors.json, what the app uses)

The labels come from the eventual sale price, so this measures whether the
tool detects a known gap, not whether any agent broke the law. That needs real
listings judged by a person -- see src/hardcases.py.
"""

import warnings

import pandas as pd

import config as C
import serve

N_SALES = 100
LEVELS = (1.0, 0.9, 0.8, 0.7)
FLAGGED = {"strong", "notable"}


def run():
    """One row per case, with the severity under each yardstick."""
    _, _, _, _, feat, meta, _ = serve.load()
    test = feat.loc[meta["test_ids"]]
    sales = test.sample(N_SALES, random_state=C.SEED)
    rows = []
    for _, r in sales.iterrows():
        area = r["land_area_m2"] if pd.notna(r["land_area_m2"]) else None
        for lvl in LEVELS:
            res = serve.value_manual(r["suburb"], r["postcode"], str(r["property_type"]),
                                     land_area_m2=area, guide=r["sale_price"] * lvl,
                                     when=r["contract_date"])
            sev_model = res["severity"]
            if "gap_vs_comps_pct" in res:
                g = res["gap_vs_comps_pct"] / res["model_typical_error_pct"]
                sev_model = next((s for k, s, _ in serve.SEVERITY if g >= k), "none")
            rows.append({"level": lvl, "type": str(r["property_type"]),
                         "model": sev_model in FLAGGED, "comps": res["severity"] in FLAGGED})
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore")
    lines = [f"# Self-test -- {N_SALES * len(LEVELS)} labelled cases", "",
             f"{N_SALES} unseen test-set sales x guides at "
             f"{', '.join(f'{l:.0%}' for l in LEVELS)} of the sale price.", "",
             "| yardstick | false alarms (guide = sale price) | caught (guide 20-30% low) |",
             "|---|---|---|"]
    df = run()
    for ys in ("model", "comps"):
        fa = df[df["level"] == 1.0][ys].mean() * 100
        caught = df[df["level"] <= 0.8][ys].mean() * 100
        lines.append(f"| {ys} | {fa:.0f}% | {caught:.0f}% |")
        print(f"    {ys:<6} false alarms {fa:.0f}%   caught {caught:.0f}%")
    (C.RESULTS / "selftest.md").write_text("\n".join(lines) + "\n")
    print(f"    saved {C.RESULTS / 'selftest.md'}")


if __name__ == "__main__":
    print(f"SELFTEST  {N_SALES * len(LEVELS)} labelled cases ...")
    main()
