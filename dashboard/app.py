"""
app.py — StegaScan Phase 5: Analyst-Facing Dashboard

A Flask web dashboard that wraps the Phase 4 detection engine
(scan_steganography + scan_backdoor + scan_weight_health +
compute_risk_score) so a non-ML-specialist security analyst can upload
or select a model, click one button, and get the same scored,
evidence-backed report Phase 4 produces on the command line — without
touching a terminal.

Design goal (per the project plan, Section 5): every finding must be
traceable to raw evidence -- the exact recovered payload text, the
exact trigger patch coordinates and collapse percentage -- not just a
numeric verdict. The dashboard surfaces that evidence directly instead
of hiding it behind a plain pass/fail badge.

Run: python3 app.py
Then open http://127.0.0.1:5000
"""

import io
import os
import sys
import time
import traceback
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, flash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.mlp import NumpyMLP
from core.scan_steganography import scan_steganography
from core.scan_backdoor import scan_backdoor
from core.scan_weight_health import scan_weight_health
from core.risk_score import compute_risk_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "stegascan-dev-key-not-for-production"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB upload cap

# Friendly labels for the four bundled Phase 2 demo models, so the
# analyst sees a plain-English description instead of a bare filename.
BUNDLED_LABELS = {
    "clean_model.npz": ("Clean baseline", "No known attack pattern embedded."),
    "stego_model.npz": ("Steganography demo", "Payload embedded in weight LSBs (EvilModel/StegoNet-style)."),
    "backdoor_model.npz": ("Backdoor demo", "Trigger-patch backdoor implanted (BadNets-style)."),
    "stego_and_backdoor.npz": ("Combined attack demo", "Both a hidden payload AND a backdoor trigger."),
}

_calibration_cache = {"X": None}


def get_calibration_set(seed=7, n=60):
    """
    The small, clean, held-out sample set the backdoor scan calibrates
    against -- the corpus-free calibration design from the project plan
    (Section 4.2): no large reference-model database required.
    Cached in-process since it's identical for every scan in this run.
    """
    if _calibration_cache["X"] is not None:
        return _calibration_cache["X"]
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    _calibration_cache["X"] = X_test[idx]
    return _calibration_cache["X"]


def list_bundled_models():
    out = []
    for fname, (label, desc) in BUNDLED_LABELS.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            out.append({"filename": fname, "label": label, "desc": desc,
                        "source": "bundled", "size_kb": round(os.path.getsize(path) / 1024, 1)})
    return out


def list_uploaded_models():
    out = []
    if os.path.isdir(UPLOAD_DIR):
        for fname in sorted(os.listdir(UPLOAD_DIR)):
            if fname.endswith(".npz"):
                path = os.path.join(UPLOAD_DIR, fname)
                out.append({"filename": fname, "label": fname, "desc": "Analyst-uploaded model.",
                            "source": "uploaded", "size_kb": round(os.path.getsize(path) / 1024, 1)})
    return out


def resolve_model_path(source, filename):
    safe_name = os.path.basename(filename)
    if source == "bundled":
        path = os.path.join(MODELS_DIR, safe_name)
    else:
        path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.abspath(path).startswith(os.path.abspath(
            MODELS_DIR if source == "bundled" else UPLOAD_DIR)):
        raise ValueError("Invalid path.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {safe_name}")
    return path


def run_full_scan(model_path):
    """Runs all three Phase 4 signals + combined risk score against one model."""
    t0 = time.time()
    model = NumpyMLP.load(model_path)
    X_calibration = get_calibration_set()

    stego_result = scan_steganography(model)
    backdoor_result = scan_backdoor(model, X_calibration)
    health_result = scan_weight_health(model)
    risk = compute_risk_score(stego_result, backdoor_result, health_result)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    n_params = sum(w.size for w in model.weights) + sum(b.size for b in model.biases)

    return {
        "risk": risk,
        "stego": stego_result,
        "backdoor": backdoor_result,
        "health": health_result,
        "elapsed_ms": elapsed_ms,
        "layer_sizes": model.layer_sizes,
        "n_params": n_params,
        "calibration_n": len(X_calibration),
    }


@app.route("/")
def index():
    bundled = list_bundled_models()
    uploaded = list_uploaded_models()
    return render_template("index.html", bundled=bundled, uploaded=uploaded)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("model_file")
    if not file or file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))
    if not file.filename.endswith(".npz"):
        flash("Only .npz weight files are accepted.", "error")
        return redirect(url_for("index"))

    safe_name = os.path.basename(file.filename)
    dest = os.path.join(UPLOAD_DIR, safe_name)
    file.save(dest)

    # Fail fast with a clear message if the file isn't a loadable model,
    # rather than letting a cryptic numpy traceback reach the analyst.
    try:
        NumpyMLP.load(dest)
    except Exception as exc:
        os.remove(dest)
        flash(f"Could not load '{safe_name}' as a StegaScan-compatible model: {exc}", "error")
        return redirect(url_for("index"))

    return redirect(url_for("report", source="uploaded", filename=safe_name))


@app.route("/report")
def report():
    source = request.args.get("source", "bundled")
    filename = request.args.get("filename", "")
    try:
        path = resolve_model_path(source, filename)
        result = run_full_scan(path)
        error = None
    except (FileNotFoundError, ValueError) as exc:
        result = None
        error = f"{exc}"
    except Exception as exc:
        result = None
        error = f"{exc}"
        traceback.print_exc()

    meta = BUNDLED_LABELS.get(filename, (filename, "Analyst-uploaded model."))
    return render_template(
        "report.html",
        filename=filename,
        source=source,
        display_label=meta[0],
        result=result,
        error=error,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("StegaScan Phase 5 — Analyst Dashboard")
    print("=" * 70)
    print(f"Bundled demo models: {len(list_bundled_models())}")
    print("Starting server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
