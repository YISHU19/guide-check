# How to run Guide Check

Start to finish: about 25 minutes, most of it unattended.

You need **Python 3.11**, an internet connection, and **~1.5 GB of free disk**.
Check your version first — 3.12+ may not have wheels for every pinned package:

```bash
python --version
```

---

## 1. Set up (2 minutes)

Clone the repo, open a terminal **inside it**, then:

(`requirements-dev.txt` rebuilds everything; `requirements.txt` alone is
enough to run the web app with `python app.py`.)

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

If PowerShell blocks the activate script, run this once and try again:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

## 2. Build everything (~20 minutes)

```bash
python src/run_all.py
```

It downloads the data itself — nothing to place by hand. Leave it running.
In a hurry? `python src/run_all.py --skip-ablation` saves about 4 minutes.

What you will see, in order:

| Stage | What it is doing | Time |
|---|---|---|
| D0 | Downloads 74 MB of NSW sales data + suburb coordinates | ~1 min |
| D1–D3 | Reads 1.5M sales, cleans to ~1.25M, adds location | ~5 min |
| D5 | Freezes the train/calibration/test split | ~10 s |
| D4 | Builds features | ~1 min |
| H1 | **Proves the scorer works using fake sabotaged results** | ~20 s |
| M1 | Baselines | ~30 s |
| M2 | Trains LightGBM and XGBoost, picks the winner | ~3 min |
| M3 | Quantile + conformal intervals | ~3 min |
| v2 | Market-drift correction | ~30 s |
| M4–M5 | Comparable-sales index, residual distribution | ~2 min |
| ABL | Five ablation retrains | ~4 min |

It stops at the first failure rather than carrying on with broken inputs.

**It will not overwrite the frozen split.** If `splits.json` already exists it
says so and moves on — that is deliberate, and it is what stops anyone
accidentally changing the test set between runs.

---

## 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8502** (or 8501). Two modes in the left sidebar:

- **Check a listing** — type in a property that is on the market now, plus its
  advertised guide. This is the real use case.
- **Validate on a past sale** — pick a sale from the frozen test set and see
  the answer against the price it actually fetched. Use this to demo that it
  works.

Stop it with `Ctrl+C`.

---

## What you should see

Results land in `results/` as timestamped Markdown. The headline numbers:

| | |
|---|---|
| Test set | 119,095 sales the model never saw, Oct 2025 – Aug 2026 |
| Typical error | **16.8%** (LightGBM) vs 29.4% for the suburb-average baseline |
| Interval | 80% claimed, **77.6%** actually achieved — reported honestly |
| Comps-anchored estimate | **13.2%** error in metro, centred at 99.9% of actual |

If your numbers differ by more than a rounding wobble, the data mirror has
refreshed since this was built, not that you broke something.

---

## Verifying the frozen split

One of the project's two hard rules is that the test split is frozen early and
never touched. `data/processed/splits_fingerprint.json` carries a SHA-256 of
each ID list. After a rebuild:

```bash
python -c "import sys; sys.path.insert(0,'src'); import split; split.fingerprint()"
```

The hashes must match the shipped file. They cannot be quietly edited.

---

## If something goes wrong

**`ModuleNotFoundError`** — the virtual environment is not active. You should
see `(.venv)` at the start of your prompt.

**Download fails or stalls** — rerun `python src/run_all.py`. Finished
downloads are kept, so it picks up from where it got to.

**Out of memory during training** — close other applications. Peak is roughly
3 GB. On a small machine, reduce `num_leaves` in `src/train_point.py`.

**App is slow to start** — first launch loads ~400 MB of artefacts. Subsequent
loads are cached and fast.

**Want to start completely clean** — delete `data/processed/`, `artefacts/` and
`predictions/`, then rerun. Keep `data/raw/psi.zip` unless you want to
re-download 74 MB.

---

## Where things are

```
src/          the pipeline, one script per stage
app/          the Streamlit app
docs/         DATA-REPORT.md -- what is actually in the source data
results/      scoring output and the ablation table
README.md     full technical write-up and limitations
```


**Read the limitations section of `README.md` before quoting any number.** In
particular: the underquoting assessment has **no validated accuracy yet** —
it needs labelled real cases (`src/hardcases.py`), and the code says so rather
than inventing a figure.
