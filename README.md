# StegaScan

**Detection of Steganographic Malware Hidden in AI Model Weights** —
a pre-deployment security scanner for the AI model supply chain.

Model hubs (Hugging Face, TensorFlow Hub, ONNX Model Zoo, internal
registries) are a software supply chain like any other, and can be
poisoned the same way. StegaScan inspects a model's raw weight file and
flags two independently-published attack classes *before* the model is
ever deployed:

1. **Steganographic payload embedding** — hiding arbitrary secret data
   (configs, keys, scripts) inside the numerical noise of trained
   weights (EvilModel, EvilModel 2.0, StegoNet).
2. **Behavioral backdoors / trojans** — a hidden trigger input forces a
   wrong output, while every normal input behaves perfectly (BadNets).

Both are demonstrated attacks in the academic literature, not a
hypothetical threat model — see `docs/research_foundations.md`.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. See the whole detection engine work end-to-end (fastest walkthrough)
python3 run_detector.py

# 2. Scan any model file yourself
python3 stegascan_cli.py models/stego_model.npz

# 3. Or use the analyst-facing web dashboard
cd dashboard && python3 app.py
# then open http://127.0.0.1:5000
```

Everything runs fully offline, no GPU, sub-second per-model scan time.

---

## Project layout

```
stegascan/
├── README.md                       <- you are here
├── requirements.txt
├── run_detector.py                 <- Phase 4 driver: sweeps the 4 labeled demo models
├── stegascan_cli.py                <- Phase 6: generic CLI, scan ANY .npz, JSON + CI exit codes
├── core/                           <- shared detection engine (Phases 1 & 4)
│   ├── mlp.py                          Phase 1: from-scratch numpy MLP
│   ├── scan_steganography.py           Phase 4: bit-depth sweep + printable-run detection
│   ├── scan_backdoor.py                Phase 4: trigger-patch grid search
│   ├── scan_weight_health.py           Phase 4: supplementary kurtosis check (low-confidence)
│   └── risk_score.py                   Phase 4: weighted 0-100 score + plain-English reasons
├── attack_sim/                     <- Phase 2: labeled attack simulation toolkit
│   ├── stego_embed.py                  LSB-substitution payload embedding
│   └── backdoor_train.py               BadNets-style trigger-patch poisoning
├── models/                         <- Phase 1 + 2 outputs: the labeled demo set
│   ├── clean_model.npz
│   ├── stego_model.npz
│   ├── backdoor_model.npz
│   ├── stego_and_backdoor.npz
│   └── ground_truth.json               exact attack parameters used for each model
├── dashboard/                      <- Phase 5: Flask analyst-facing web UI
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── uploads/
└── docs/
    ├── phase4_detection_engine.md
    ├── phase5_dashboard.md
    └── research_foundations.md
```

---

## 5-minute live demo script

Written for a timed judging slot. Swap in whichever surface (CLI or
dashboard) fits the room.

**0:00 – 0:30 — Frame the threat.** "Model hubs are a software supply
chain. We built a scanner that opens a model's weight file before
deployment and checks for two real, published attacks: hidden payloads
and behavioral backdoors — without needing the original training
pipeline."

**0:30 – 1:30 — Run the sweep.** `python3 run_detector.py` — narrate
while it prints: four labeled models (clean / stego / backdoor / both),
each scored 0–100, cross-checked against ground truth live in the
terminal. Land on: *"8 out of 8 correct — zero false positives, zero
false negatives."*

**1:30 – 3:00 — Show the evidence, not just the score.** Open the
dashboard (`cd dashboard && python3 app.py`), click into
`stego_and_backdoor.npz`. Point at:
- The actual recovered payload string, not a "malware detected" badge.
- The actual winning trigger patch (corner, size, brightness) and its
  100% collapse rate against the calibration set.
- The weight-health tag explicitly labeled LOW CONFIDENCE — show the
  judges the tool is honest about what it doesn't trust.

**3:00 – 4:00 — Prove the pivot, not just the result.** "We first tried
raw bit-entropy as a stego signal and it failed — trained weights are
already near-maximal entropy in their low bits. We tested that,
rejected it, and pivoted to active extraction. Same story for backdoor
detection: output-layer weight-norm outliers didn't separate clean from
tampered, so we moved to a trigger-patch grid search instead." This is
the single strongest credibility signal in the demo — say it explicitly.

**4:00 – 4:45 — Deployability.** `python3 stegascan_cli.py --dir models
--fail-on ELEVATED` — show the non-zero exit code. "This is one flag
away from being a pre-commit hook or a model-registry-upload gate."

**4:45 – 5:00 — Honest limitations, stated proactively.** "This scan
catches contiguous LSB substitution and corner-patch triggers — the
patterns in the published literature we built against. A pre-encrypted
payload or a spatially distributed trigger would need additional
detection layers. A LOW score means no known pattern was found, not a
safety certificate. That's not a gap we're hiding — it's the same
caveat every cited defense paper makes about itself."

---

## Evaluation criteria mapping

| Criterion | How this project addresses it |
|---|---|
| Cybersecurity Effectiveness | Two independently validated, empirically demonstrated detection signals against two real, published attack classes |
| Steganography Awareness | Full-tensor LSB scan and bit-depth sweep are directly grounded in the EvilModel/StegoNet/steganographic-capacity literature |
| Technical Soundness | Rejected a weak signal (raw bit-entropy) after testing it, rather than shipping it; every heuristic has a demonstrated true positive against a labeled attack model |
| Explainability | Every finding surfaces concrete evidence — recovered text, exact trigger coordinates, collapse percentage — not just a score |
| Practical Feasibility | Runs fully offline, no GPU, sub-second scan time; working CLI, driver script, and web dashboard all included |
| Clarity of Demo | Four labeled models (clean / stego / backdoor / both) give an unambiguous before/after story; `ground_truth.json` makes every claim checkable |

---

## Honest limitations

Stated proactively, not as a defensive afterthought — this is a
deliberate part of how the project is scoped:

- **Toy-scale model and dataset**, chosen for demo clarity and
  reproducible, offline execution — not a claim of production
  readiness on real-world architectures (transformers, large CNNs).
- **`scan_steganography` is tuned to contiguous LSB substitution** (the
  pattern used by EvilModel/StegoNet-style attacks). Non-contiguous bit
  placement or a pre-encrypted payload (indistinguishable from noise
  even after correct extraction) would evade this check — a documented
  gap, not a claimed-solved problem. The format/framework breadth item
  in the roadmap (ONNX, SafeTensors, PyTorch checkpoints) is unbuilt.
- **`scan_backdoor` searches a bounded, guessable trigger space**
  (corner patches up to 3×3 against a small calibration set). A
  spatially distributed or learned trigger could evade the grid; the
  Neural-Cleanse-style gradient-based optimization described in the
  roadmap is the documented next step, not yet implemented.
- **`scan_weight_health` is explicitly not a validated detector.** It
  contributes a small, non-decisive nudge to the score and is labeled
  LOW CONFIDENCE everywhere it appears — console output, dashboard, and
  JSON — because a related weight-statistic signal (output-layer norm
  z-scores) was tested and rejected in Phase 3.
- **A LOW risk score means "no known attack pattern was found,"** not a
  certification of safety — consistent with how every cited defense
  paper frames its own guarantees.
- **The dashboard's `app.secret_key` and dev server are not
  production-hardened** — Flask's built-in server is explicitly
  unsuitable for production deployment; this is a demo/analyst tool,
  not a hardened public-facing service.
- **Not yet built (roadmap only):** the Neural-Cleanse-style optimized
  trigger search, a spectral/activation-based secondary backdoor
  detector, a STRIP-style runtime companion, a real CI/CD Action or Hub
  webhook (the CLI's `--fail-on` exit codes are a first step, not the
  full integration), and a live "embed it on stage" demo mode.

---

## What's built vs. roadmap, at a glance

| Phase | Status |
|---|---|
| 0 — Threat modeling & research grounding | ✅ Built (see `docs/research_foundations.md`) |
| 1 — Baseline transparent model | ✅ Built (`core/mlp.py`) |
| 2 — Attack simulation toolkit | ✅ Built (`attack_sim/`, `models/`) |
| 3 — Empirical signal validation | ✅ Done — findings folded into Phase 4's thresholds and documented in `docs/phase4_detection_engine.md` |
| 4 — Core detection engine | ✅ Built (`core/scan_*.py`, `core/risk_score.py`, `run_detector.py`) |
| 5 — Analyst-facing dashboard | ✅ Built (`dashboard/`) |
| 6 — Evaluation, packaging & demo script | ✅ This document + `stegascan_cli.py` |
| 4.4 — Neural-Cleanse-style trigger optimization | 🔲 Roadmap |
| 4.5 — Spectral/activation-based secondary detector | 🔲 Roadmap |
| 4.6 — STRIP-style runtime companion | 🔲 Roadmap |
| 4.7 — Full CI/CD gate (GitHub Action / Hub webhook) | 🔲 Roadmap (CLI exit codes are a first step) |
| 4.8 — ONNX / SafeTensors / PyTorch format support | 🔲 Roadmap |
| 4.9 — Live "attack it on stage" demo | 🔲 Roadmap (optional) |
