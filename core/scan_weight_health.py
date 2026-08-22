"""
scan_weight_health.py — Detection engine module: scan_weight_health()

A SUPPLEMENTARY, explicitly LOW-CONFIDENCE check. Phase 3 already showed
that a related idea (output-layer weight-NORM outliers) fails outright
as a backdoor detector -- an innocent class in the clean model scored a
higher outlier z-score than the real backdoored class in the tampered
model. This module is not a resurrection of that rejected signal; it's
a much more general "does this tensor's value distribution look
statistically unusual" check (excess kurtosis per tensor), kept in the
engine only as a minor corroborating data point, weighted low
(+8 out of 100) in the combined risk score, and never used to flag a
model on its own.

Rationale for keeping it at all: fine-tuning (as in backdoor implanting)
and gross tampering can sometimes leave a heavier- or lighter-tailed
weight distribution than typical trained weights. It is NOT reliable
enough to trust standalone -- that is the explicit, stated conclusion of
Phase 3 -- so it is surfaced to the analyst as a footnote, not a
verdict.
"""

import numpy as np


# Normally-trained float32 weight tensors typically show mild
# platykurtic-to-mesokurtic excess kurtosis. This is a loose reference
# band, not a validated threshold -- unlike the Phase 3 signals, this
# module has NOT been shown to reliably separate clean from tampered.
EXPECTED_EXCESS_KURTOSIS_BAND = (-1.5, 3.0)


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64).reshape(-1)
    mean = x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0
    m4 = np.mean((x - mean) ** 4)
    return float(m4 / (std ** 4) - 3.0)


def scan_weight_health(model) -> dict:
    """
    Computes excess kurtosis for every weight tensor and flags any tensor
    falling outside a loose reference band. LOW CONFIDENCE: this signal
    was not validated to reliably separate clean/tampered models in
    Phase 3 (a related weight-statistic signal was explicitly rejected)
    and should never be the sole basis for a verdict.

    Returns:
      flagged: bool (True if ANY tensor falls outside the reference band)
      per_tensor: {tensor_name: excess_kurtosis}
      outlier_tensors: list of tensor names outside the band
      reasons: list of plain-English strings, explicitly caveated as low-confidence
    """
    per_tensor = {}
    outlier_tensors = []
    lo, hi = EXPECTED_EXCESS_KURTOSIS_BAND

    for i, W in enumerate(model.weights):
        k = _excess_kurtosis(W)
        per_tensor[f"W{i}"] = k
        if not (lo <= k <= hi):
            outlier_tensors.append(f"W{i}")

    flagged = len(outlier_tensors) > 0

    reasons = []
    if flagged:
        reasons.append(
            f"[LOW CONFIDENCE] Tensor(s) {outlier_tensors} show excess kurtosis "
            f"outside the loose reference band {EXPECTED_EXCESS_KURTOSIS_BAND}. "
            f"This is a weak, unvalidated corroborating signal only -- see "
            f"Phase 3 finding that a related weight-statistic check failed to "
            f"reliably separate clean from tampered models. Do not act on this "
            f"alone."
        )
    else:
        reasons.append(
            "[LOW CONFIDENCE] No tensors show unusual weight-distribution "
            "kurtosis. This check is weak corroborating evidence only, not a "
            "clearance."
        )

    return {
        "flagged": flagged,
        "per_tensor": per_tensor,
        "outlier_tensors": outlier_tensors,
        "reasons": reasons,
    }
