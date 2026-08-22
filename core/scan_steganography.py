"""
scan_steganography.py — Detection engine module: scan_steganography()

Production version of the signal validated in Phase 3
(validate_signal_extraction.py). Generalized from "only check the first
layer" to sweeping EVERY weight and bias tensor in the model, since a
real attacker isn't obligated to pick the tensor our demo happened to
target in Phase 2.

Method: for each tensor, for each bit_depth in 1..8, extract the low
bit_depth bits of every element (flattened, row-major) as a byte stream
and find the longest contiguous run of printable ASCII. Report the best
(tensor, bit_depth) combination found across the whole model.

Threshold (MIN_FLAG_RUN_LEN=20) is carried over from Phase 3's empirical
finding: clean models produced coincidental printable runs up to 9 bytes
long; embedded payloads produced 280+ byte runs. 20 sits with real
margin above the observed noise ceiling.
"""

import numpy as np

MIN_FLAG_RUN_LEN = 20   # empirically set in Phase 3 -- see phase3/README.md
BIT_DEPTHS_TO_SWEEP = range(1, 9)


def _low_bits_bytes(tensor: np.ndarray, bit_depth: int) -> bytes:
    flat = tensor.reshape(-1).view(np.uint32)
    mask = np.uint32((1 << bit_depth) - 1)
    vals = flat & mask
    bitstring = "".join(format(int(v), f"0{bit_depth}b") for v in vals)
    n_bytes = len(bitstring) // 8
    return bytes(int(bitstring[i * 8:(i + 1) * 8], 2) for i in range(n_bytes))


def _longest_printable_run(data: bytes):
    best_start, best_len = -1, 0
    cur_start, cur_len = -1, 0
    for i, b in enumerate(data):
        if 0x20 <= b <= 0x7E:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    if best_len == 0:
        return -1, b""
    return best_start, data[best_start:best_start + best_len]


def _named_tensors(model):
    """Every weight AND bias tensor in the model, named for reporting."""
    tensors = []
    for i, W in enumerate(model.weights):
        tensors.append((f"W{i}", W))
    for i, b in enumerate(model.biases):
        # bias vectors are float32 too and just as embeddable, though
        # much lower capacity -- still worth scanning.
        tensors.append((f"b{i}", b))
    return tensors


def scan_steganography(model) -> dict:
    """
    Sweeps every tensor in `model` at every bit-depth 1-8, looking for a
    long contiguous printable-ASCII run in the extracted low bits.

    Returns a dict:
      flagged: bool
      tensor: name of the tensor with the strongest hit
      bit_depth: bit depth of the strongest hit
      run_length: length in bytes of the longest printable run found
      recovered_text: the actual recovered text (decoded, best-effort)
      all_hits: list of every (tensor, bit_depth, run_length) with run_length > 0,
                sorted descending -- useful for the analyst report
      reasons: list of plain-English explanation strings
    """
    all_hits = []
    best = {"tensor": None, "bit_depth": None, "run_length": 0, "recovered_text": ""}

    for tensor_name, tensor in _named_tensors(model):
        for depth in BIT_DEPTHS_TO_SWEEP:
            data = _low_bits_bytes(tensor, depth)
            _, run = _longest_printable_run(data)
            if len(run) > 0:
                all_hits.append((tensor_name, depth, len(run)))
            if len(run) > best["run_length"]:
                best = {
                    "tensor": tensor_name,
                    "bit_depth": depth,
                    "run_length": len(run),
                    "recovered_text": run.decode("ascii", errors="replace"),
                }

    all_hits.sort(key=lambda h: -h[2])
    flagged = best["run_length"] >= MIN_FLAG_RUN_LEN

    reasons = []
    if flagged:
        preview = best["recovered_text"][:80]
        reasons.append(
            f"Recovered a {best['run_length']}-byte printable-ASCII run from "
            f"tensor '{best['tensor']}' at bit_depth={best['bit_depth']}: "
            f"{preview!r}{'...' if len(best['recovered_text']) > 80 else ''}"
        )
        reasons.append(
            f"This exceeds the empirical clean-model noise ceiling "
            f"(observed max 9 bytes across reference models, "
            f"flag threshold set at {MIN_FLAG_RUN_LEN} bytes)."
        )
    else:
        reasons.append(
            f"No printable run >= {MIN_FLAG_RUN_LEN} bytes found in any tensor "
            f"at any bit-depth 1-8 (best: {best['run_length']} bytes in "
            f"'{best['tensor']}', consistent with ordinary noise)."
        )

    return {
        "flagged": flagged,
        "tensor": best["tensor"],
        "bit_depth": best["bit_depth"],
        "run_length": best["run_length"],
        "recovered_text": best["recovered_text"],
        "all_hits": all_hits[:10],  # top 10 for the report, avoid clutter
        "reasons": reasons,
    }
