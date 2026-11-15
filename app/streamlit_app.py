"""Guide Check -- the application.

Two modes:
  Check a listing   type in a LIVE listing and its advertised guide. This is
                    the real use case: nobody needs a valuation for a house
                    that already sold.
  Validate          pick a past sale the model never saw, and see the answer
                    against the price it actually fetched. For demonstrating
                    that the thing works.

The comparable sales lead. They ARE the argument -- the number is a summary of
them, and NSW underquoting law is written around comparable evidence, not
around a model's opinion.

Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config as C          # noqa: E402
import serve                # noqa: E402
import smsf_pdf             # noqa: E402

st.set_page_config(page_title="Guide Check", page_icon="GC", layout="wide")
INK = "#14432f"

st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem; max-width: 1180px;}}
  .gc-hero {{border-left: 4px solid {INK}; padding: .1rem 0 .1rem 1rem; margin-bottom: 1rem;}}
  .gc-hero h1 {{margin:0; font-size:1.65rem; color:{INK};}}
  .gc-hero p  {{margin:.25rem 0 0; color:#555; font-size:.9rem;}}
  .gc-card {{background:#e8f1ec; border:1px solid #9db8ab; border-radius:8px;
             padding:.9rem 1.1rem; margin-bottom:.6rem;}}
  .gc-warn {{background:#fff4e6; border:1px solid #b08968; border-radius:8px;
             padding:.9rem 1.1rem;}}
  .gc-num  {{font-size:1.9rem; font-weight:700; color:{INK}; line-height:1.15;}}
  .gc-sub  {{color:#555; font-size:.82rem;}}
  .gc-pct  {{font-size:3rem; font-weight:700; line-height:1;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gc-hero">
  <h1>Guide Check</h1>
  <p>Is this price guide defensible against comparable sales? &mdash; plus a
     calibrated valuation and an SMSF evidence pack.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model artefacts ...")
def boot():
    import json
    serve.load()
    return pd.read_parquet(C.FEAT_PQ), json.loads(C.SPLITS.read_text())


feat, meta = boot()
suburbs = sorted(feat["suburb"].astype(str).unique())

mode = st.sidebar.radio("Mode", ["Check a listing", "Validate on a past sale"])

# ------------------------------------------------------------------ input ----
res = subject = None
if mode == "Check a listing":
    with st.sidebar:
        st.subheader("The listing")
        st.caption("Type in a property that is on the market now.")
        addr = st.text_input("Address", "12 Example Street")
        sub = st.selectbox("Suburb", suburbs,
                           index=suburbs.index("MUDGEE") if "MUDGEE" in suburbs else 0)
        pc = st.text_input("Postcode", "2850")
        ptype = st.radio("Type", ["house", "unit"], horizontal=True)
        area = st.number_input("Land area (m2, 0 if unknown)", 0, 100_000, 600, 10)
        c1, c2 = st.columns(2)
        beds = c1.number_input("Beds", 0, 10, 3)
        baths = c2.number_input("Baths", 0, 10, 1)
        st.caption("Beds and baths are recorded but cannot affect the estimate "
                   "yet -- they do not exist in the NSW sales data.")
        st.divider()
        guide = st.number_input("Advertised guide ($)", 0, 50_000_000,
                                500_000, 25_000)
        go = st.button("Assess", type="primary", use_container_width=True)
    if go or True:
        try:
            res = serve.value_manual(sub, pc, ptype,
                                     land_area_m2=area or None, address=addr,
                                     guide=guide or None,
                                     bedrooms=beds or None, bathrooms=baths or None)
            subject = res["subject"]
        except ValueError as e:
            st.error(str(e))
            st.stop()
else:
    test = feat.loc[meta["test_ids"]]
    with st.sidebar:
        st.subheader("Past sale")
        st.caption("From the frozen test set. The model never saw these.")
        sub = st.selectbox("Suburb", sorted(test["suburb"].astype(str).unique()))
        pool = test[test["suburb"].astype(str) == sub]
        ptype = st.radio("Type", sorted(pool["property_type"].astype(str).unique()),
                         horizontal=True)
        pool = pool[pool["property_type"].astype(str) == ptype].head(60)
        labels = [f"{r.address}  ({r.contract_date:%b %Y})" for r in pool.itertuples()]
        if not labels:
            st.stop()
        pick = st.selectbox("Property", labels)
        key = pool.index[labels.index(pick)]
        row = feat.loc[key]
        st.divider()
        guide = st.number_input("Pretend guide ($)", 0, 50_000_000,
                                int(row["sale_price"] * .8 // 25_000 * 25_000), 25_000)
    res = serve.value(row, guide=guide or None, key=key)
    subject = {k: row[k] for k in ("address", "suburb", "postcode",
                                   "property_type", "land_area_m2")}
    actual = float(row["sale_price"])

st.markdown(f"**{subject['address']}, {subject['suburb']} {subject['postcode']}**"
            f" &nbsp;·&nbsp; {subject['property_type']} &nbsp;·&nbsp; "
            + (f"{subject['land_area_m2']:,.0f} m&sup2;"
               if subject.get("land_area_m2") and pd.notna(subject["land_area_m2"])
               else "land area not given"), unsafe_allow_html=True)

# ------------------------------------------------- the guide assessment ------
if "comps_percentile" in res:
    pct = res["comps_percentile"]
    sev = res["severity"]
    col = {"strong": "#b04a28", "notable": "#b08968",
           "mild": "#7a7a4a", "none": INK}.get(sev, INK)
    box = "gc-warn" if sev in ("strong", "notable") else "gc-card"
    a, b = st.columns([1, 2.6])
    with a:
        st.markdown(f"""<div class="{box}" style="text-align:center;">
          <div class="gc-sub">Guide sits at the</div>
          <div class="gc-pct" style="color:{col};">{pct:.0f}<span
            style="font-size:1.1rem;">th</span></div>
          <div class="gc-sub">percentile of comparable sales</div>
        </div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""<div class="{box}">
          <b>{res['band'].capitalize()}</b><br>
          <span class="gc-sub">
          Of {res['comps_total']} comparable sales adjusted for differences,
          <b>{res['comps_below']}</b> sold below this guide of
          <b>${res['guide']:,.0f}</b>.
          Comparable evidence centres on <b>${res['comps_estimate']:,.0f}</b>
          ({res['gap_vs_comps_pct']:+.0f}% vs the guide).<br><br>
          Separately, the model puts the chance this property is worth more than
          the guide at <b>{res['prob_worth_more_pct']:.0f}%</b>.<br><br>
          {res['disclaimer']}</span>
        </div>""", unsafe_allow_html=True)
    st.write("")

# -------------------------------------------------------- comps lead ---------
st.subheader("Comparable sales")
st.caption("Recent sales of similar properties nearby, each adjusted for how it "
           "differs from the subject. This is the evidence; everything else "
           "summarises it.")
if res["comps_estimate"]:
    cmp = res["comps"].copy()
    cmp["contract_date"] = cmp["contract_date"].dt.strftime("%d %b %Y")
    show = pd.DataFrame({
        "Address": cmp["address"], "Suburb": cmp["suburb"],
        "Land m2": cmp["land_area_m2"].map(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "-"),
        "Sold": cmp["contract_date"],
        "Price": cmp["sale_price"].map("${:,.0f}".format),
        "Adjustment": cmp["adjustment_pct"].map("{:+.0f}%".format),
        "Adjusted": cmp["adjusted_price"].map("${:,.0f}".format)})
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(f"Estimate from comparable evidence: "
               f"**${res['comps_estimate']:,.0f}** "
               f"(middle half of {res['comps_n']} adjusted comparables spans "
               f"{res['comps_spread_pct']:.0f}% of that).")

st.divider()
left, right = st.columns([1, 1])
with left:
    st.subheader("Model valuation")
    cov = int(res["nominal_coverage"] * 100)
    st.markdown(f"""<div class="gc-card">
      <div class="gc-sub">Point estimate</div>
      <div class="gc-num">${res['point']:,.0f}</div>
      <div class="gc-sub" style="margin-top:.5rem;">Calibrated {cov}% range</div>
      <div style="font-size:1.15rem;font-weight:600;color:{INK};">
        ${res['lower']:,.0f} &ndash; ${res['upper']:,.0f}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="gc-warn">
      <b>How much to trust this</b><br>
      <span class="gc-sub">
      For {res['segment']} properties our estimates are typically within
      <b>{res['typical_error_pct']:.0f}%</b>, measured on
      {meta['n_calib']:,} held-out sales. The {cov}% range actually contained
      the true price <b>77.6%</b> of the time on the test set, so it is
      slightly narrow. Treat this as a prompt to ask questions, not an answer.
      </span></div>""", unsafe_allow_html=True)
    if mode != "Check a listing":
        inside = res["lower"] <= actual <= res["upper"]
        st.markdown(f"""<div class="gc-card" style="margin-top:.6rem;">
          <div class="gc-sub">Actual sale price (held out)</div>
          <div style="font-size:1.3rem;font-weight:700;color:{INK};">
            ${actual:,.0f}</div>
          <div class="gc-sub">model error
            {(res['point']-actual)/actual*100:+.1f}% &nbsp;·&nbsp;
            {'inside' if inside else 'OUTSIDE'} the range &nbsp;·&nbsp;
            comps estimate error
            {((res['comps_estimate'] or np.nan)-actual)/actual*100:+.1f}%</div>
        </div>""", unsafe_allow_html=True)

with right:
    st.subheader("What drove the estimate")
    st.caption("SHAP contributions, in log-price space.")
    st.bar_chart(pd.DataFrame(
        {"contribution": res["drivers"].head(7).sort_values()}),
        horizontal=True, color="#2c5a44", height=280)
    if "bed_bath_note" in res:
        st.info(res["bed_bath_note"], icon="i")

st.divider()
st.subheader("SMSF evidence pack")
st.caption("A Reg 8.02B workpaper: property, estimate, range, comparable sales, "
           "method note and date.")
st.download_button("Download SMSF evidence PDF",
                   smsf_pdf.build(subject, res),
                   file_name=f"smsf-evidence-{subject['suburb']}.pdf".replace(" ", "-"),
                   mime="application/pdf", type="primary")

st.caption(
    f"LightGBM on {meta['n_train']:,} NSW sales to "
    f"{meta['temporal_cutoffs']['train_end']}; conformal calibration on "
    f"{meta['n_calib']:,}; evaluated on {meta['n_test']:,} held-out sales. "
    f"Split frozen {meta['freeze_date']}, blocked by SA2 and time. "
    "The guide assessment has no validated accuracy yet -- it needs labelled "
    "cases (src/hardcases.py).")
