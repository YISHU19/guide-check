"""Guide Check web app: the site in public/ plus two API endpoints that run the
real trained model (src/serve.py).

    pip install -r requirements.txt
    python app.py                     # http://localhost:5000

On Vercel the pages in public/ are served from the CDN and only /api/* reaches
this Flask app; locally Flask serves both.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import serve  # noqa: E402

PUBLIC = ROOT / "public"
PAGES = {"check", "how-it-works", "how-to-use", "limits"}

app = Flask(__name__, static_folder=str(PUBLIC), static_url_path="")

_suburbs_cache = None
_places_cache = None


def _suburbs():
    global _suburbs_cache
    if _suburbs_cache is None:
        feat = serve.load()[4]
        _suburbs_cache = sorted(feat["suburb"].astype(str).unique().tolist())
    return _suburbs_cache


def _places():
    """{suburb: [postcodes]}, most-sold postcode first, for linking the two
    form fields."""
    global _places_cache
    if _places_cache is None:
        feat = serve.load()[4]
        pairs = feat[["suburb", "postcode"]].astype(str).value_counts()
        places = {}
        for (suburb, postcode), _ in pairs.items():
            places.setdefault(suburb, []).append(postcode)
        _places_cache = dict(sorted(places.items()))
    return _places_cache


def _clean(obj):
    """Make serve.py's output JSON-safe: numpy scalars/arrays, pandas Series
    and DataFrames all need converting."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.reset_index(drop=True).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _clean(obj.to_dict())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


@app.get("/")
def index():
    return send_from_directory(PUBLIC, "index.html")


@app.get("/<page>")
def page(page):
    if page not in PAGES:
        abort(404)
    return send_from_directory(PUBLIC, f"{page}.html")


@app.get("/api/suburbs")
def api_suburbs():
    return jsonify(_suburbs())


@app.get("/api/places")
def api_places():
    return jsonify(_places())


@app.post("/api/check")
def api_check():
    body = request.get_json(force=True) or {}
    try:
        guide = body.get("guide")
        result = serve.value_manual(
            suburb=body["suburb"],
            postcode=body.get("postcode", "0000"),
            property_type=body.get("property_type", "house"),
            land_area_m2=body.get("land_area_m2"),
            address=body.get("address", "(entered manually)"),
            guide=float(guide) if guide not in (None, "") else None,
            bedrooms=body.get("bedrooms"),
            bathrooms=body.get("bathrooms"),
        )
        return jsonify(_clean(result))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Something went wrong scoring that property."}), 500


if __name__ == "__main__":
    print("Loading model...")
    serve.load()
    print(f"Ready. {len(_suburbs())} NSW suburbs indexed.")
    app.run(host="0.0.0.0", port=5000, debug=False)
