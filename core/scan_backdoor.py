"""
scan_backdoor.py — Detection engine module: scan_backdoor()

Production version of the signal validated in Phase 3
(validate_signal_trigger_grid.py). Sweeps a grid of candidate trigger
patches against a small clean calibration set and looks for a patch
that forces anomalous prediction collapse onto a single class.

Informed by Neural Cleanse's core insight: a genuinely backdoored label
requires an anomalously small, consistent perturbation to force, versus
the natural variability of legitimate labels. This is a hackathon-scale
simplification -- a bounded, guessable grid of corner patches -- not the
gradient-based per-label optimization Neural Cleanse actually uses (that
upgrade is documented as Phase 4.4 in the project roadmap).

Threshold (COLLAPSE_FLAG_THRESHOLD=0.6) is carried over from Phase 3's
empirical finding: the clean model's best-of-grid decoy patch collapsed
26.7% of predictions; the true trigger on a backdoored model collapsed
100%. 0.6 sits with wide margin above the observed clean-model ceiling.
"""

import numpy as np

CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]
PATCH_SIZES = [1, 2, 3]
PATCH_VALUES = [0.0, 0.5, 1.0]
COLLAPSE_FLAG_THRESHOLD = 0.6  # empirically set in Phase 3 -- see phase3/README.md


def _apply_patch(images, corner, size, value):
    imgs = images.reshape(-1, 8, 8).copy()
    if corner == "top_left":
        imgs[:, :size, :size] = value
    elif corner == "top_right":
        imgs[:, :size, -size:] = value
    elif corner == "bottom_left":
        imgs[:, -size:, :size] = value
    elif corner == "bottom_right":
        imgs[:, -size:, -size:] = value
    return imgs.reshape(-1, 64)


def scan_backdoor(model, X_calibration: np.ndarray) -> dict:
    """
    Sweeps a grid of candidate trigger patches against `X_calibration`
    (a small clean, held-out sample set the caller provides -- this is
    the "corpus-free calibration" design: no large reference-model
    database needed, just a handful of legitimate task inputs).

    Returns a dict:
      flagged: bool
      corner, patch_size, patch_value: the winning patch parameters
      target_class: the class predictions collapsed onto
      collapse_fraction: fraction of calibration set forced to target_class
      grid_summary: top 5 grid results, sorted by collapse fraction, for the report
      reasons: list of plain-English explanation strings
    """
    grid_results = []
    for corner in CORNERS:
        for size in PATCH_SIZES:
            for value in PATCH_VALUES:
                patched = _apply_patch(X_calibration, corner, size, value)
                preds = model.predict(patched)
                counts = np.bincount(preds, minlength=10)
                collapse_class = int(np.argmax(counts))
                collapse_frac = float(counts[collapse_class] / len(preds))
                grid_results.append({
                    "corner": corner, "patch_size": size, "patch_value": value,
                    "target_class": collapse_class, "collapse_fraction": collapse_frac,
                })

    grid_results.sort(key=lambda r: -r["collapse_fraction"])
    top = grid_results[0]
    flagged = top["collapse_fraction"] >= COLLAPSE_FLAG_THRESHOLD

    reasons = []
    if flagged:
        reasons.append(
            f"A {top['patch_size']}x{top['patch_size']} patch at the "
            f"{top['corner'].replace('_', ' ')} corner (brightness={top['patch_value']}) "
            f"forces {top['collapse_fraction']*100:.1f}% of calibration predictions "
            f"to class {top['target_class']}, regardless of the input's true content."
        )
        reasons.append(
            f"This exceeds the empirical clean-model noise ceiling "
            f"(observed max 26.7% best-of-grid collapse on a reference clean model, "
            f"flag threshold set at {COLLAPSE_FLAG_THRESHOLD*100:.0f}%)."
        )
    else:
        reasons.append(
            f"No candidate patch in the {len(grid_results)}-combination grid forced "
            f"collapse >= {COLLAPSE_FLAG_THRESHOLD*100:.0f}% "
            f"(best: {top['collapse_fraction']*100:.1f}% via {top['corner']}/"
            f"size={top['patch_size']}/value={top['patch_value']}, "
            f"consistent with natural class variability)."
        )

    return {
        "flagged": flagged,
        "corner": top["corner"],
        "patch_size": top["patch_size"],
        "patch_value": top["patch_value"],
        "target_class": top["target_class"],
        "collapse_fraction": top["collapse_fraction"],
        "grid_summary": grid_results[:5],
        "reasons": reasons,
    }
