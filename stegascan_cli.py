#!/usr/bin/env python3
"""
stegascan_cli.py — Phase 6: general-purpose CLI batch scanner.

Unlike run_detector.py (Phase 4's driver, hardcoded to the four bundled
labeled demo models), this CLI scans ANY StegaScan-compatible .npz
weight file(s) the user points it at -- a single file, several files,
or every .npz in a directory -- and is meant to be the fast terminal
walkthrough for a live demo, or a first building block toward the
CI/CD supply-chain gate described in the project roadmap (Section 4.7):
the --fail-on flag below returns a non-zero exit code exactly the way a
pre-commit hook or CI step would need.

Usage:
    python3 stegascan_cli.py models/stego_model.npz
    python3 stegascan_cli.py models/*.npz
    python3 stegascan_cli.py --dir models/
    python3 stegascan_cli.py --dir models/ --json
    python3 stegascan_cli.py --dir models/ --fail-on ELEVATED

Exit codes:
    0   every scanned model came in below --fail-on's threshold
    1   at least one scanned model met or exceeded --fail-on's threshold
    2   a model file could not be loaded / scanned (I/O or format error)
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from core.mlp import NumpyMLP
from core.scan_steganography import scan_steganography
from core.scan_backdoor import scan_backdoor
from core.scan_weight_health import scan_weight_health
from core.risk_score import compute_risk_score

BAND_ORDER = {"LOW": 0, "ELEVATED": 1, "HIGH": 2}

_calibration_cache = {"X": None}


def get_calibration_set(seed=7, n=60):
    """
    Corpus-free calibration set (project plan Section 4.2): a small,
    clean, held-out sample of the task's own data, not a large external
    reference-model database. Cached once per CLI invocation.
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


def resolve_targets(args):
    targets = []
    if args.dir:
        targets.extend(sorted(glob.glob(os.path.join(args.dir, "*.npz"))))
    targets.extend(args.files)
    # de-dupe while preserving order
    seen = set()
    out = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def scan_one(path):
    t0 = time.time()
    model = NumpyMLP.load(path)
    X_calibration = get_calibration_set()

    stego_result = scan_steganography(model)
    backdoor_result = scan_backdoor(model, X_calibration)
    health_result = scan_weight_health(model)
    risk = compute_risk_score(stego_result, backdoor_result, health_result)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return {
        "file": path,
        "risk_score": risk["risk_score"],
        "risk_band": risk["risk_band"],
        "contributions": risk["contributions"],
        "reasons": risk["reasons"],
        "stego_flagged": stego_result["flagged"],
        "backdoor_flagged": backdoor_result["flagged"],
        "health_flagged": health_result["flagged"],
        "recovered_text": stego_result["recovered_text"] if stego_result["flagged"] else None,
        "backdoor_trigger": (
            {
                "corner": backdoor_result["corner"],
                "patch_size": backdoor_result["patch_size"],
                "patch_value": backdoor_result["patch_value"],
                "target_class": backdoor_result["target_class"],
                "collapse_fraction": backdoor_result["collapse_fraction"],
            }
            if backdoor_result["flagged"] else None
        ),
        "elapsed_ms": elapsed_ms,
    }


def print_human(result, verbose):
    band = result["risk_band"]
    print(f"\n{'='*70}")
    print(f"{result['file']}")
    print(f"{'='*70}")
    print(f"RISK SCORE: {result['risk_score']}/100  [{band}]   ({result['elapsed_ms']} ms)")
    print(f"  steganography: +{result['contributions']['steganography']}   "
          f"backdoor: +{result['contributions']['backdoor']}   "
          f"weight_health: +{result['contributions']['weight_health']}")
    if verbose:
        print("\nEvidence:")
        for r in result["reasons"]:
            print(f"  - {r}")


def print_summary_table(results):
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    width = max((len(r["file"]) for r in results), default=10)
    for r in results:
        print(f"{r['file']:<{width}}  score={r['risk_score']:>3}/100  [{r['risk_band']:<8}]  "
              f"stego={'Y' if r['stego_flagged'] else 'n'}  "
              f"backdoor={'Y' if r['backdoor_flagged'] else 'n'}")


def main():
    parser = argparse.ArgumentParser(
        description="StegaScan CLI — scan AI model weight files for hidden "
                     "steganographic payloads and backdoor triggers before deployment."
    )
    parser.add_argument("files", nargs="*", help="One or more .npz model files to scan.")
    parser.add_argument("--dir", metavar="PATH", help="Scan every .npz file in this directory.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text.")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary table, not per-file evidence.")
    parser.add_argument(
        "--fail-on", metavar="BAND", choices=["LOW", "ELEVATED", "HIGH"], default=None,
        help="Exit with a non-zero status if any scanned model's risk band meets or "
             "exceeds this threshold (e.g. --fail-on ELEVATED). Intended for use as a "
             "pre-commit / CI gate step -- see project roadmap Section 4.7.",
    )
    args = parser.parse_args()

    targets = resolve_targets(args)
    if not targets:
        parser.error("No model files given. Pass file paths, or use --dir to scan a directory.")

    results = []
    had_error = False
    for path in targets:
        try:
            result = scan_one(path)
        except Exception as exc:
            had_error = True
            if args.json:
                results.append({"file": path, "error": str(exc)})
            else:
                print(f"\n[ERROR] Could not scan '{path}': {exc}", file=sys.stderr)
            continue
        results.append(result)
        if not args.json and not args.quiet:
            print_human(result, verbose=True)

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.quiet or len(results) > 1:
        print_summary_table([r for r in results if "error" not in r])

    exit_code = 0
    if had_error:
        exit_code = 2
    elif args.fail_on:
        threshold = BAND_ORDER[args.fail_on]
        if any(BAND_ORDER[r["risk_band"]] >= threshold for r in results if "error" not in r):
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
