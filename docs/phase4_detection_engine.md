# StegaScan — Phase 4: Core Detection Engine

Turns the two signals validated in Phase 3 into a combined, explainable
scanning tool. Every finding traces to concrete evidence — recovered
text, exact trigger coordinates, collapse percentages — never just an
opaque number.

## What's here

| File | Purpose |
|---|---|
| `core/scan_steganography.py` | `scan_steganography(model)` — sweeps every tensor at bit-depths 1-8, flags long printable-ASCII runs. |
| `core/scan_backdoor.py` | `scan_backdoor(model, X_calibration)` — 36-combination trigger-patch grid search, flags anomalous prediction collapse. |
| `core/scan_weight_health.py` | `scan_weight_health(model)` — supplementary, explicitly low-confidence kurtosis outlier check. |
| `core/risk_score.py` | `compute_risk_score(...)` — weighted combination into a 0–100 score with plain-English reasons. |
| `run_detector.py` | Driver: runs the full engine against all four Phase 2 models and validates against `models/ground_truth.json`. |

Run it from the project root: `python3 run_detector.py`

## Design: weighting reflects validated confidence, not just plan spec

| Signal | Weight | Phase 3 status |
|---|---|---|
| Steganography (`scan_steganography`) | **+55** | Validated — 9 bytes noise vs 280+ bytes payload |
| Backdoor (`scan_backdoor`) | **+40** | Validated — 26.7% noise vs 100% real trigger |
| Weight health (`scan_weight_health`) | **+8** | **Not** validated — explicitly low-confidence, supplementary only |

The two validated signals dominate the score (95/103 possible points).
The weight-health check can only ever nudge a score, never singlehandedly
push a clean model into a flagged bracket.

Thresholds are carried over unchanged from Phase 3's empirical
measurements, not re-guessed:
- `scan_steganography`: flag threshold = 20-byte printable run (Phase 3
  observed clean noise ceiling was 9 bytes)
- `scan_backdoor`: flag threshold = 60% prediction collapse (Phase 3
  observed clean noise ceiling was 26.7%)

## Generalizations beyond the Phase 3 prototype scripts

- **Steganography scan checks every tensor**, not just the first
  layer (`W0`) that Phase 2's demo happened to target. A real attacker
  isn't obligated to pick the tensor our own demo used — `W0`...`Wn` and
  `b0`...`bn` are all swept.
- **Backdoor scan takes the calibration set as a parameter** rather than
  hardcoding the dataset — the corpus-free calibration design (Section
  4.2 of the project plan): any small clean, held-out sample set works,
  no large reference-model database required.

## Full sweep results (last run)

| Model | Risk Score | Band | Stego flag | Backdoor flag | Ground truth match |
|---|---|---|---|---|---|
| `clean_model.npz` | 0/100 | LOW | False | False | ✅ |
| `stego_model.npz` | 55/100 | HIGH | True | False | ✅ |
| `backdoor_model.npz` | 40/100 | ELEVATED | False | True | ✅ |
| `stego_and_backdoor.npz` | 95/100 | HIGH | True | True | ✅ |

**Detection accuracy: 8/8 correct flags (100%)** across all four labeled
models — zero false positives, zero false negatives, cross-checked
against `models/ground_truth.json` from Phase 2.

## Honest limitations (unchanged from earlier phases, restated here)

- `scan_steganography` is tuned to contiguous LSB substitution. A
  non-contiguous bit placement or a pre-encrypted payload (indistinguishable
  from noise even after correct extraction) would not produce a printable
  run and would evade this check — documented gap, not claimed solved.
- `scan_backdoor` searches a bounded, guessable trigger space (corner
  patches up to 3x3). A spatially distributed or learned trigger could
  evade the grid — the Neural-Cleanse-style gradient-based optimization
  (project roadmap Section 4.4) is the documented next step.
- `scan_weight_health` is explicitly not a validated detector. It is
  surfaced as a minor corroborating data point only, and this is stated
  directly in its own output, not just in this doc.
- A LOW risk score means "no known attack pattern was found," not a
  certification of safety.
