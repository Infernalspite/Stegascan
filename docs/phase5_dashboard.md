# StegaScan — Phase 5: Analyst-Facing Dashboard

Wraps the Phase 4 detection engine in a Flask web dashboard so a
non-ML-specialist security analyst can upload or select a model, click
one button, and get the same scored, evidence-backed report Phase 4
produces on the command line — without touching a terminal. Mirrors the
"clinician-facing" usability bar from the original challenge brief,
adapted to a security-analyst persona.

## Run it

```bash
cd dashboard
pip install flask numpy scikit-learn --break-system-packages   # if not already installed
python3 app.py
```

Then open **http://127.0.0.1:5000**. Everything runs locally — no data
leaves the machine, consistent with the project's offline-first design.

## What's here

| Path | Purpose |
|---|---|
| `app.py` | Flask app: case index, upload handler, report route. Imports the Phase 4 engine unchanged from `../core/`. |
| `templates/` | `index.html` (case index / upload), `report.html` (scored case report), `base.html` (shared layout). |
| `templates/index.html` | Case index — pick one of the four bundled Phase 2 demo models, or upload a new `.npz`. |
| `templates/report.html` | Scored case report — ink-stamp verdict, per-signal evidence tags, recovered payload text, trigger grid table. |
| `static/style.css` | Design system (see below). |
| `models/` | The four labeled Phase 2 demo models + `ground_truth.json`, bundled so the dashboard has something to show on first run. |
| `uploads/` | Analyst-uploaded models land here (created at runtime, empty in the repo). |

The dashboard does **not** reimplement any detection logic — it imports
`scan_steganography`, `scan_backdoor`, `scan_weight_health`, and
`compute_risk_score` directly from the shared `core/` package also used
by Phase 4's CLI and Phase 6's batch scanner, so the verdict an analyst
sees in the browser is guaranteed to match the terminal output for the
same file.

## Design: a case file, not a dashboard

The brief for this page: a security analyst under time pressure needs
to trust a verdict on an artifact they can't just "look at" the way
they could look at a suspicious email attachment. The visual language
borrows from a physical evidence case file rather than a generic
metrics dashboard, because the tool's whole value proposition is
*traceable evidence*, not a clean number:

- **Manila case-folder palette** (warm tan paper, ink-dark text) instead
  of a SaaS-dashboard cream/terracotta or dark/neon scheme.
- **Ink-stamp verdict** — the LOW / ELEVATED / HIGH risk band renders
  like a rotated rubber stamp, the way a physical case file would be
  marked, rather than a colored badge.
- **Evidence tags** — each of the three signals (steganography,
  backdoor, weight-health) is shown as a punched, tagged card, echoing
  a physical evidence tag rather than a dashboard "card component."
- **System-stack typography only** — a typewriter-style monospace for
  headings, a plain sans for body copy, a mono face for recovered
  payload text and tensor tables. No webfonts are loaded, so the
  dashboard keeps working with zero network access, in step with the
  project's own "runs fully offline" claim.

## Explainability, not just a badge

Every finding traces to raw evidence, shown directly on the page:

- The **actual recovered payload text** (not just "steganography
  detected") for a stego flag, plus the top 5 (tensor, bit-depth,
  run-length) hits from the sweep.
- The **actual winning trigger patch** — corner, size, brightness,
  target class, collapse percentage — plus the top 5 grid results, for
  a backdoor flag.
- Per-tensor excess-kurtosis values for the (explicitly low-confidence)
  weight-health check.

This directly targets the Explainability judging criterion: an analyst
never has to trust an opaque score without being able to see exactly
what triggered it.

## Honest limitations carried into the UI

The report page's "Reading this report" panel restates, in the
analyst's context, the same caveats documented since Phase 0/3/4: a LOW
score is "no known pattern found," not a safety certification; the
steganography check is tuned to contiguous LSB substitution; the
backdoor check searches a bounded corner-patch grid. Nothing here
overclaims beyond what Phase 3 actually validated.

## What comes next (not part of this phase)

Phase 6 packages the whole project (Phases 1–5) for judging: a
generic CLI batch-scanner that works on arbitrary uploaded `.npz`
files (not just the four bundled demo models), a timed live-demo
script, and a final offline-ready zip.
