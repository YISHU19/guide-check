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

# Severity is scaled by the segment's measured error, not by percentile alone.
# Percentile bands (strong below the 5th, notable below the 15th) only fired on
# discounts nobody needed a model to notice: the middle half of the adjusted
# comparables spans ~28% of their median, so a real 15% underquote landed at
# the 24th percentile. A gap is judged by how many typical errors wide it is.
SEVERITY = [            # (min gap in typical errors, severity, wording)
    (2.0, "strong", "well below the comparable evidence"),
    (1.0, "notable", "below the comparable evidence"),
    (0.5, "mild", "at the low end of the comparable evidence"),
]
CONSISTENT = ("none", "consistent with the comparable evidence")

# Readable names for the SHAP drivers shown in the app.
DRIVER_NAMES = {
    "suburb": "suburb", "land_area_m2": "land area", "log_area": "land area (log)",
    "has_area": "land area recorded", "zoning": "zoning",
    "days_since_start": "date of sale", "sale_year": "year of sale",
    "sale_month": "month of sale", "suburb_median_logprice": "suburb price level",
    "suburb_n_sales": "suburb sales volume", "sa2_code": "statistical area",
    "postcode": "postcode", "property_type": "property type",
    "dist_cbd_km": "distance to CBD", "lat": "latitude", "lon": "longitude",
}
N_TOP_DRIVERS = 4


@lru_cache(maxsize=1)
def load():
    point = joblib.load(C.ART / "model_lgbm.pkl")
    iv = joblib.load(C.ART / "interval_v2.pkl")
    art_comps = joblib.load(C.ART / "comps_index.pkl")
    resid = joblib.load(C.ART / "residuals.pkl")
    feat = pd.read_parquet(C.FEAT_PQ)
    # The split is only needed to score or rebuild, not to serve, so the
    # deployed app does not have to ship the 61 MB splits.json.
    meta = json.loads(C.SPLITS.read_text()) if C.SPLITS.exists() else None

    # LightGBM's own TreeSHAP (pred_contrib) gives exactly the values
    # shap.TreeExplainer does -- max difference 0.0 over 200 rows -- without
    # pulling shap and numba into the serving bundle.
    def explain(X):
        return point.predict(X, num_iteration=point.best_iteration,
                             pred_contrib=True)[:, :-1]
    return point, iv, art_comps, resid, feat, meta, explain


@lru_cache(maxsize=1)
def comps_errors():
    """Typical error of the comps estimate for a typed-in property, by segment
    and property type, measured on unseen sales by src/comps_errors.py."""
    p = C.ART / "comps_errors.json"
    return json.loads(p.read_text())["median_ape_pct"] if p.exists() else {}


def _prep(row, feat):
    X = pd.DataFrame([row])[FEATURES]
    for c in X.columns:
        if str(feat[c].dtype) == "category":
            X[c] = pd.Categorical(X[c], categories=feat[c].cat.categories)
    return X


def _segment(row):
    d = row.get("dist_cbd_km")
    return "metro" if d is not None and np.isfinite(d) and d < 50 else "regional"


def _driver_value(feature, row, as_at):
    v = row.get(feature)
    if feature == "days_since_start":
        return as_at.strftime("%Y-%m")
    if feature == "land_area_m2":
        return f"{v:,.0f} m2" if v is not None and np.isfinite(v) else "not recorded"
    if feature == "suburb_median_logprice":
        return f"{v:.2f}"
    if feature == "dist_cbd_km":
        return f"{v:.1f} km"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _evidence(cutoff, as_at):
    age = int((as_at - pd.Timestamp(cutoff)).days)
    months = round(age / 30.44)
    if months < 3:
        note = (f"The comparable sales run to {cutoff}, close to the date being "
                "valued.")
    else:
        note = (f"The comparable sales end {cutoff}, {months} months before the "
                "date being valued. Prices have moved since. Treat the estimate "
                "as a floor on today's evidence, and rebuild the index (python "
                "src/comps.py) against fresher PSI data before relying on it.")
    return age, note


def value(row, guide=None, key=None, as_at=None):
    """Assess one property. `row` carries every model feature; `as_at` is the
    date being valued (defaults to the date implied by the row)."""
    point, iv, art_comps, resid, feat, meta, explain = load()
    X = _prep(row, feat)
    if as_at is None:
        as_at = feat["contract_date"].min() + pd.Timedelta(days=int(row["days_since_start"]))
    as_at = pd.Timestamp(as_at).normalize()

    log_pt = float(point.predict(X, num_iteration=point.best_iteration)[0]) + iv["delta"]
    log_lo = float(iv["m_lo"].predict(X, num_iteration=iv["m_lo"].best_iteration)[0]) \
        + iv["delta"] - iv["qhat_v2"]
    log_hi = float(iv["m_hi"].predict(X, num_iteration=iv["m_hi"].best_iteration)[0]) \
        + iv["delta"] + iv["qhat_v2"]
    est, lo, hi = np.exp(log_pt), np.exp(log_lo), np.exp(log_hi)
    lo, hi = min(lo, est), max(hi, est)

    seg = _segment(row)
    r = resid["resid_by_segment"].get(seg, resid["resid_all"])
    model_err = float(np.median(np.abs(np.exp(r) - 1)) * 100)
    # The guide is judged against the comps estimate, so it is that estimate's
    # error -- larger than the model's for a typed-in property -- that sets
    # the scale. Falls back to the model's own error if it was never measured.
    typical_err = comps_errors().get(seg, {}).get(str(row["property_type"]), model_err)

    # --- comps-anchored estimate (the better-centred one) -------------------
    ce = CE.estimate(row, point, iv, art_comps, feat, key=key)

    drivers = (pd.Series(explain(X).reshape(-1), index=FEATURES)
                 .sort_values(key=np.abs, ascending=False))
    top = [{"feature": DRIVER_NAMES.get(f, f), "value": _driver_value(f, row, as_at),
            "effect_pct": float((np.exp(v) - 1) * 100)}
           for f, v in drivers.head(N_TOP_DRIVERS).items()]

    age, note = _evidence(art_comps["cutoff"], as_at)
    out = {
        "point": float(est), "lower": float(lo), "upper": float(hi),
        "nominal_coverage": C.NOMINAL_COVERAGE,
        "interval_width_pct": float((hi - lo) / est * 100),
        "segment": seg, "typical_error_pct": typical_err,
        "model_typical_error_pct": model_err,
        "drivers": drivers, "top_drivers": top,
        "comps": ce["table"] if ce else comps_mod.query(art_comps, row,
                                                        exclude_key=key),
        "comps_estimate": ce["estimate"] if ce else None,
        "comps_distribution": ce["distribution"] if ce else None,
        "comps_n": ce["n_comps"] if ce else 0,
        "comps_spread_pct": ce["spread_pct"] if ce else None,
        "comps_cutoff": art_comps["cutoff"],
        "comps_lookback_days": comps_mod.LOOKBACK_DAYS,
        "evidence_age_days": age, "evidence_note": note,
        "empirical_coverage_pct": 77.6,
        "empirical_coverage_note": (
            "Measured coverage on the frozen test set is 77.6% against a "
            "nominal 80%; the band is slightly narrow. See results/ for the "
            "full evaluation."),
    }

    if guide is not None and np.isfinite(guide) and guide > 0:
        out.update(_assess_guide(guide, est, r, ce, typical_err))
    return out


def _assess_guide(guide, est, resid_seg, ce, typical_err):
    """Assess an advertised guide against the evidence -- as a percentile.

    Two independent readings, because they answer different questions:

      vs comparable sales  -- the legal standard. What share of adjusted
                              comparable sales sit BELOW this guide.
      vs the model         -- calibrated from held-out residuals: the estimated
                              probability the property is worth more than the
                              guide.

    Severity comes from the gap to the comps estimate in units of the
    segment's typical error, so a gap inside the noise is never flagged.
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
        gap = float((ce["estimate"] - guide) / ce["estimate"] * 100)
        d["gap_vs_comps_pct"] = gap
        d["gap_in_typical_errors"] = gap / typical_err
        d["severity"], d["band"] = CONSISTENT
        for k, sev, band in SEVERITY:
            if d["gap_in_typical_errors"] >= k:
                d["severity"], d["band"] = sev, band
                break
        side = "below" if gap >= 0 else "above"
        d["verdict"] = (
            f"This guide sits at the {pct:.0f}th percentile of "
            f"{ce['n_comps']} comparable sales adjusted for differences, "
            f"{abs(gap):.0f}% {side} their median ({d['band']}). Typical error "
            f"for this segment is {typical_err:.1f}%.")
    else:
        d["verdict"] = ("Not enough comparable sales to assess this guide "
                        "against evidence.")
        d["severity"] = "unknown"

    d["disclaimer"] = (
        "A statistical observation about price against comparable sales. It is "
        "not an allegation of underquoting, which depends on the agent's own "
        "reasonable estimate at the time of listing, which no price model can "
        "observe.")
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

    res = value(pd.Series(row), guide=guide, as_at=when)
    res["subject"] = {"address": address, "suburb": suburb,
                      "postcode": postcode, "property_type": property_type,
                      "land_area_m2": row["land_area_m2"],
                      "bedrooms": bedrooms, "bathrooms": bathrooms,
                      "valued_as_at": when.strftime("%Y-%m-%d"),
                      "location_basis": (
                          "Suburb-level estimate: features are taken from the "
                          f"most recent recorded sale in {suburb}. Two "
                          "properties in this suburb with the same type and "
                          "land area get the same estimate. Street, aspect and "
                          "condition are not modelled.")}
    if bedrooms or bathrooms:
        res["bed_bath_note"] = (
            "Bedroom and bathroom counts were recorded but could not be used: "
            "they are absent from the NSW sales data the model was trained on. "
            "They are the single largest missing feature; collecting them "
            "from live listings is what would let a future model use them.")
    return res
