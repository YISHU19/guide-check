"""The serving layer: turns a subject property into a complete assessment.

Two entry points:

  value(row, ...)          an existing record from the feature table
  value_manual(...)        a LIVE listing typed in by address -- what the tool
                           actually needs to do, since nobody needs a valuation
                           for a house that already sold

What changed, and why
---------------------
The underquoting output used to be a binary flag: guide < lower_bound x 0.95.
Measured on the frozen test set, that caught a real 20% underquote only 1 time
in 4 while wrongly flagging 1 honest guide in 15. Narrowing the population did
not help (119k rows down to 17.7k moved the catch rate 25.5% -> 26.1%).

The problem was the target, not the accuracy. "Will this sell for more than the
guide?" is the wrong question -- properties routinely sell above their guide in
a hot auction without anyone underquoting. NSW law asks something different and
narrower: is the guide a reasonable estimate given COMPARABLE SALES? So we now
assess the guide against the comparable evidence, and report a percentile
rather than a verdict.
"""

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

import comps as comps_mod
import comps_estimator as CE
import config as C
from features import FEATURES

# Percentile bands for the wording. Deliberately not a single threshold: the
# whole point of moving off the binary flag is that the evidence is continuous.
BAND_STRONG = 5.0     # below ~all comparable evidence
BAND_NOTABLE = 15.0   # below most of it


@lru_cache(maxsize=1)
def load():
    point = joblib.load(C.ART / "model_lgbm.pkl")
    iv = joblib.load(C.ART / "interval_v2.pkl")
    art_comps = joblib.load(C.ART / "comps_index.pkl")
    resid = joblib.load(C.ART / "residuals.pkl")
    feat = pd.read_parquet(C.FEAT_PQ)
    meta = json.loads(C.SPLITS.read_text())
    import shap
    return point, iv, art_comps, resid, feat, meta, shap.TreeExplainer(point)


def _prep(row, feat):
    X = pd.DataFrame([row])[FEATURES]
    for c in X.columns:
        if str(feat[c].dtype) == "category":
            X[c] = pd.Categorical(X[c], categories=feat[c].cat.categories)
    return X


def _segment(row):
    d = row.get("dist_cbd_km")
    return "metro" if d is not None and np.isfinite(d) and d < 50 else "regional"


def value(row, guide=None, key=None):
    """Assess one property. `row` carries every model feature."""
    point, iv, art_comps, resid, feat, meta, explainer = load()
    X = _prep(row, feat)

    log_pt = float(point.predict(X, num_iteration=point.best_iteration)[0]) + iv["delta"]
    log_lo = float(iv["m_lo"].predict(X, num_iteration=iv["m_lo"].best_iteration)[0]) \
        + iv["delta"] - iv["qhat_v2"]
    log_hi = float(iv["m_hi"].predict(X, num_iteration=iv["m_hi"].best_iteration)[0]) \
        + iv["delta"] + iv["qhat_v2"]
    est, lo, hi = np.exp(log_pt), np.exp(log_lo), np.exp(log_hi)
    lo, hi = min(lo, est), max(hi, est)

    seg = _segment(row)
    r = resid["resid_by_segment"].get(seg, resid["resid_all"])
    typical_err = float(np.median(np.abs(np.exp(r) - 1)) * 100)

    # --- comps-anchored estimate (the better-centred one) -------------------
    ce = CE.estimate(row, point, iv, art_comps, feat, key=key)

    sv = explainer.shap_values(X)
    drivers = (pd.Series(np.asarray(sv).reshape(-1), index=FEATURES)
                 .sort_values(key=np.abs, ascending=False))

    out = {
        "point": float(est), "lower": float(lo), "upper": float(hi),
        "nominal_coverage": C.NOMINAL_COVERAGE,
        "interval_width_pct": float((hi - lo) / est * 100),
        "segment": seg, "typical_error_pct": typical_err,
        "drivers": drivers,
        "comps": ce["table"] if ce else comps_mod.query(art_comps, row,
                                                        exclude_key=key),
        "comps_estimate": ce["estimate"] if ce else None,
        "comps_distribution": ce["distribution"] if ce else None,
        "comps_n": ce["n_comps"] if ce else 0,
        "comps_spread_pct": ce["spread_pct"] if ce else None,
        "empirical_coverage_note": (
            "Measured coverage on the frozen test set is 77.6% against a "
            "nominal 80% -- the band is slightly narrow, see results/."),
    }

    if guide is not None and np.isfinite(guide) and guide > 0:
        out.update(_assess_guide(guide, est, r, ce))
    return out


def _assess_guide(guide, est, resid_seg, ce):
    """Assess an advertised guide against the evidence -- as a percentile.

    Two independent readings, because they answer different questions:

      vs comparable sales  -- the legal standard. What share of adjusted
                              comparable sales sit BELOW this guide.
      vs the model         -- calibrated from held-out residuals: the estimated
                              probability the property is worth more than the
                              guide.
    """
    d = dict(guide=float(guide))

    # Probability the true value exceeds the guide, from measured residuals.
    # P(est * exp(r) > guide) = fraction of residuals with r > log(guide/est).
    p_above = float((resid_seg > np.log(guide / est)).mean() * 100)
    d["prob_worth_more_pct"] = p_above
    d["model_percentile"] = 100.0 - p_above

    if ce is not None:
        pct = CE.guide_percentile(ce["distribution"], guide)
        d["comps_percentile"] = pct
        n_below = int(round(pct / 100 * ce["n_comps"]))
        d["comps_below"] = n_below
        d["comps_total"] = ce["n_comps"]
        d["gap_vs_comps_pct"] = float((ce["estimate"] - guide)
                                      / ce["estimate"] * 100)
        if pct <= BAND_STRONG:
            d["band"] = "below almost all comparable evidence"
            d["severity"] = "strong"
        elif pct <= BAND_NOTABLE:
            d["band"] = "below most comparable evidence"
            d["severity"] = "notable"
        elif pct <= 40:
            d["band"] = "at the low end of comparable evidence"
            d["severity"] = "mild"
        else:
            d["band"] = "consistent with comparable evidence"
            d["severity"] = "none"
        d["verdict"] = (
            f"This guide sits at the {pct:.0f}th percentile of "
            f"{ce['n_comps']} comparable sales adjusted for differences -- "
            f"{d['band']}.")
    else:
        d["verdict"] = ("Not enough comparable sales to assess this guide "
                        "against evidence.")
        d["severity"] = "unknown"

    d["disclaimer"] = (
        "A statistical observation about price against comparable sales. It is "
        "not an allegation of underquoting, which depends on the agent's own "
        "reasonable estimate at the time of listing -- something no price model "
        "can observe.")
    return d


# --------------------------------------------------------------- live input --
def value_manual(suburb, postcode, property_type, land_area_m2=None,
                 address="(entered manually)", guide=None,
                 bedrooms=None, bathrooms=None, when=None):
    """Assess a LIVE listing typed in by hand.

    bedrooms/bathrooms are accepted and echoed back but CANNOT yet influence
    the estimate: they do not exist anywhere in the NSW PSI source data, so the
    model was never trained on them. They are collected here so the listing
    dataset is being built from day one -- see data/reference/listing_template.csv.
    """
    _, _, _, _, feat, meta, _ = load()
    suburb = str(suburb).strip().upper()
    postcode = str(postcode).strip().zfill(4)

    ref = feat[feat["suburb"].astype(str) == suburb]
    if ref.empty:
        raise ValueError(f"Suburb '{suburb}' not found in the NSW sales data.")
    base = ref.iloc[-1]                       # most recent sale in that suburb

    when = pd.Timestamp(when) if when is not None else pd.Timestamp.today()
    t0 = feat["contract_date"].min()

    row = {c: base[c] for c in FEATURES}
    row.update({
        "property_type": property_type,
        "postcode": postcode if postcode in set(
            feat["postcode"].astype(str)) else base["postcode"],
        "land_area_m2": float(land_area_m2) if land_area_m2 else np.nan,
        "log_area": (np.log(float(land_area_m2))
                     if land_area_m2 and float(land_area_m2) > 0 else np.nan),
        "has_area": 1 if land_area_m2 else 0,
        "sale_year": int(when.year), "sale_month": int(when.month),
        "days_since_start": int((when - t0).days),
    })

    res = value(pd.Series(row), guide=guide)
    res["subject"] = {"address": address, "suburb": suburb,
                      "postcode": postcode, "property_type": property_type,
                      "land_area_m2": row["land_area_m2"],
                      "bedrooms": bedrooms, "bathrooms": bathrooms}
    if bedrooms or bathrooms:
        res["bed_bath_note"] = (
            "Bedroom and bathroom counts were recorded but could not be used: "
            "they are absent from the NSW sales data the model was trained on. "
            "They are the single largest missing feature -- collecting them "
            "from live listings is what would let a future model use them.")
    return res
