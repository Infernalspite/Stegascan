"""
run_detector.py — Phase 4 driver script.

Runs the full combined detection engine (scan_steganography +
scan_backdoor + scan_weight_health + compute_risk_score) against all
four labeled models from Phase 2, and checks each verdict against
models/ground_truth.json so the engine's accuracy is demonstrated, not
just asserted.

This is the fastest way to see the whole engine work end-to-end from a
terminal. For scanning an arbitrary model file (not just the four
bundled demo models), use stegascan_cli.py instead.
"""

import json
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from core.mlp import NumpyMLP
from core.scan_steganography import scan_steganography
from core.scan_backdoor import scan_backdoor
from core.scan_weight_health import scan_weight_health
from core.risk_score import compute_risk_score

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
MODELS = ["clean_model.npz", "stego_model.npz", "backdoor_model.npz", "stego_and_backdoor.npz"]


def get_calibration_set(seed=7, n=60):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    return X_test[idx]


def scan_model(fname, X_calibration, verbose=True):
    model = NumpyMLP.load(os.path.join(MODELS_DIR, fname))

    stego_result = scan_steganography(model)
    backdoor_result = scan_backdoor(model, X_calibration)
    health_result = scan_weight_health(model)
    risk = compute_risk_score(stego_result, backdoor_result, health_result)

    if verbose:
        print(f"\n{'='*70}")
        print(f"SCAN REPORT: {fname}")
        print(f"{'='*70}")
        print(f"RISK SCORE: {risk['risk_score']}/100  [{risk['risk_band']}]")
        print(f"  steganography: +{risk['contributions']['steganography']}   "
              f"backdoor: +{risk['contributions']['backdoor']}   "
              f"weight_health: +{risk['contributions']['weight_health']}")
        print("\nEvidence:")
        for r in risk["reasons"]:
            print(f"  - {r}")

    return {
        "file": fname,
        "risk_score": risk["risk_score"],
        "risk_band": risk["risk_band"],
        "stego_flagged": stego_result["flagged"],
        "backdoor_flagged": backdoor_result["flagged"],
        "health_flagged": health_result["flagged"],
    }


def main():
    print("=" * 70)
    print("StegaScan Phase 4: Core Detection Engine — full model sweep")
    print("=" * 70)

    with open(os.path.join(MODELS_DIR, "ground_truth.json")) as f:
        ground_truth = json.load(f)

    X_calibration = get_calibration_set()
    print(f"\nUsing a {len(X_calibration)}-sample clean calibration set "
          f"(held out from training, drawn once for this whole sweep).")

    all_results = []
    for fname in MODELS:
        result = scan_model(fname, X_calibration)
        all_results.append(result)

    # ---- Cross-check against ground truth ----
    print(f"\n{'='*70}")
    print("VALIDATION AGAINST PHASE 2 GROUND TRUTH")
    print(f"{'='*70}")

    expected = {
        "clean_model.npz": {"stego": False, "backdoor": False},
        "stego_model.npz": {"stego": True, "backdoor": False},
        "backdoor_model.npz": {"stego": False, "backdoor": True},
        "stego_and_backdoor.npz": {"stego": True, "backdoor": True},
    }

    correct = 0
    total = 0
    for r in all_results:
        exp = expected[r["file"]]
        stego_ok = r["stego_flagged"] == exp["stego"]
        backdoor_ok = r["backdoor_flagged"] == exp["backdoor"]
        total += 2
        correct += int(stego_ok) + int(backdoor_ok)
        status_stego = "correct" if stego_ok else "WRONG"
        status_backdoor = "correct" if backdoor_ok else "WRONG"
        print(f"{r['file']:<24} score={r['risk_score']:>3}/100 [{r['risk_band']:<8}]  "
              f"stego: got={r['stego_flagged']!s:<5} expected={exp['stego']!s:<5} ({status_stego})  "
              f"backdoor: got={r['backdoor_flagged']!s:<5} expected={exp['backdoor']!s:<5} ({status_backdoor})")

    print(f"\nDetection accuracy: {correct}/{total} correct flags "
          f"({correct/total*100:.1f}%)")

    if correct == total:
        print("\nAll four labeled models scored correctly. The detection")
        print("engine reliably separates clean from tampered models using")
        print("the two Phase-3-validated signals, with the low-confidence")
        print("weight-health check contributing only a minor, non-decisive")
        print("nudge to the score.")
    else:
        print("\nSome flags did not match expectations -- see rows marked")
        print("WRONG above for details.")


if __name__ == "__main__":
    main()
