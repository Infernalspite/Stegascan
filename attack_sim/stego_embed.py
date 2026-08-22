"""
stego_embed.py — Steganographic payload embedding module (Phase 2).

Implements full-tensor least-significant-bit (LSB) substitution on a
model's float32 weight tensor, directly following the embedding pattern
described in:

  - StegoNet (Liu et al., ACSAC 2020) — LSB modification of model weights
  - "Steganographic Capacity of Selected/Deep Learning Models" (arXiv
    2306.17189 / 2308.15502) — full-tensor (not selectively-targeted)
    LSB substitution across every weight in a layer

This is the pattern our Phase 4 detector's bit-depth sweep is built to
find. (EvilModel's *selective*, redundant-weight-targeting variant is a
related but distinct technique — noted in the README as a documented
gap.)

IMPORTANT: The "payload" embedded here is a short, clearly-labeled demo
string (see train_attacks.py) — plain ASCII text used purely to prove
the embedding/extraction pipeline works end-to-end. This module embeds
arbitrary bytes; it does not contain or distribute any actual malicious
executable code. Its purpose is to produce a labeled test case for a
defensive scanner, the same way antivirus vendors use the inert EICAR
string to test AV engines without shipping real malware.
"""

import numpy as np


def _bytes_to_bitstring(payload: bytes) -> str:
    """Convert bytes to a string of '0'/'1' characters, MSB-first per byte,
    with a 32-bit length header so extraction knows where to stop."""
    length_header = format(len(payload), "032b")
    body = "".join(format(byte, "08b") for byte in payload)
    return length_header + body


def _bitstring_to_bytes(bits: str) -> bytes:
    n = len(bits) // 8
    return bytes(int(bits[i * 8:(i + 1) * 8], 2) for i in range(n))


def capacity_bits(tensor: np.ndarray, bit_depth: int) -> int:
    """How many payload bits this tensor can hold at the given bit depth
    (excluding the 32-bit length header)."""
    return tensor.size * bit_depth - 32


def embed_payload(tensor: np.ndarray, payload: bytes, bit_depth: int = 2) -> np.ndarray:
    """
    Embed `payload` bytes into the low `bit_depth` mantissa bits of every
    float32 element in `tensor`, flattened in row-major order.

    Returns a NEW array (does not mutate the input) with the same shape
    and dtype as `tensor`.
    """
    if tensor.dtype != np.float32:
        raise ValueError("Embedding target must be float32")
    if not (1 <= bit_depth <= 8):
        raise ValueError("bit_depth must be between 1 and 8")

    bitstring = _bytes_to_bitstring(payload)
    cap = capacity_bits(tensor, bit_depth)
    if len(bitstring) - 32 > cap:
        raise ValueError(
            f"Payload too large: needs {len(bitstring) - 32} bits, "
            f"tensor capacity is {cap} bits at bit_depth={bit_depth}. "
            f"Increase bit_depth or choose a larger tensor."
        )

    flat = tensor.reshape(-1).view(np.uint32).copy()
    mask = np.uint32((1 << bit_depth) - 1)          # e.g. bit_depth=2 -> 0b11
    clear_mask = np.uint32(~mask & 0xFFFFFFFF)        # clears the low bits

    n_values_needed = -(-len(bitstring) // bit_depth)  # ceil division
    if n_values_needed > flat.size:
        raise ValueError("Payload requires more elements than tensor provides")

    # Pad bitstring to a multiple of bit_depth so every chunk is full width
    padded_bits = bitstring + "0" * ((-len(bitstring)) % bit_depth)

    chunks = [
        int(padded_bits[i:i + bit_depth], 2)
        for i in range(0, len(padded_bits), bit_depth)
    ]
    chunk_arr = np.array(chunks, dtype=np.uint32)

    flat[: len(chunk_arr)] = (flat[: len(chunk_arr)] & clear_mask) | chunk_arr

    stego_tensor = flat.view(np.float32).reshape(tensor.shape)
    return stego_tensor


def extract_payload(tensor: np.ndarray, bit_depth: int) -> bytes:
    """
    Inverse of embed_payload: read the low `bit_depth` bits of every
    element (row-major), reconstruct the bitstring, read the 32-bit
    length header, and return exactly that many payload bytes.

    Included here (rather than only in the Phase 4 detector) so this
    module is independently testable / round-trip-verifiable.
    """
    flat = tensor.reshape(-1).view(np.uint32)
    mask = np.uint32((1 << bit_depth) - 1)

    header_chunks_needed = -(-32 // bit_depth)
    header_vals = flat[:header_chunks_needed] & mask
    header_bits = "".join(format(int(v), f"0{bit_depth}b") for v in header_vals)[:32]
    payload_len_bytes = int(header_bits, 2)
    payload_len_bits = payload_len_bytes * 8

    total_bits_needed = 32 + payload_len_bits
    total_chunks_needed = -(-total_bits_needed // bit_depth)

    vals = flat[:total_chunks_needed] & mask
    full_bits = "".join(format(int(v), f"0{bit_depth}b") for v in vals)
    payload_bits = full_bits[32:32 + payload_len_bits]

    return _bitstring_to_bytes(payload_bits)


if __name__ == "__main__":
    # Self-test: round-trip a demo payload through a random tensor
    rng = np.random.default_rng(0)
    dummy = (rng.standard_normal((32, 16)) * 0.3).astype(np.float32)
    msg = b"StegaScan Phase 2 self-test payload OK"
    for depth in (1, 2, 4):
        stego = embed_payload(dummy, msg, bit_depth=depth)
        recovered = extract_payload(stego, bit_depth=depth)
        status = "PASS" if recovered == msg else "FAIL"
        print(f"bit_depth={depth}: recovered={recovered!r} [{status}]")
