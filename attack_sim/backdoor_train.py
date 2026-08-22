"""
backdoor_train.py — Behavioral backdoor / trojan module (Phase 2).

Implements a simplified version of the dirty-label data-poisoning attack
from BadNets (Gu, Liu, Dolan-Gavitt & Garg, IEEE Access 2019): a small
fixed trigger pattern is stamped onto a subset of training images, those
images are relabeled to a fixed attacker-chosen target class, and the
model is fine-tuned on a mix of clean + poisoned data.

Result: the model performs normally (~clean accuracy) on every ordinary
input, but any input containing the trigger patch gets forced to the
target class regardless of its true content — directly analogous to
BadNets' backdoored street-sign classifier (stop sign + sticker ->
speed-limit sign).

This is a benign research simulation on a toy digit-classification model
— it demonstrates the *mechanism* of the attack on synthetic data so a
detector can be validated against a known ground truth, not a real
deployed system.
"""

import os
import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.mlp import NumpyMLP


TRIGGER_TARGET_CLASS = 0     # attacker-chosen: any triggered input -> predicted "0"
TRIGGER_PATCH_SIZE = 2       # 2x2 corner block
TRIGGER_CORNER = "bottom_right"
TRIGGER_VALUE = 1.0          # max brightness (post-normalization) corner patch
POISON_FRACTION = 0.15       # fraction of training set that gets poisoned


def apply_trigger(images: np.ndarray, patch_size=TRIGGER_PATCH_SIZE,
                   corner=TRIGGER_CORNER, value=TRIGGER_VALUE) -> np.ndarray:
    """
    Stamp the fixed trigger patch onto a batch of flattened 8x8 digit
    images (shape (N, 64), normalized 0-1). Returns a new array.
    """
    imgs = images.reshape(-1, 8, 8).copy()
    if corner == "bottom_right":
        imgs[:, -patch_size:, -patch_size:] = value
    elif corner == "top_left":
        imgs[:, :patch_size, :patch_size] = value
    else:
        raise ValueError(f"Unknown corner: {corner}")
    return imgs.reshape(-1, 64)


def one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def build_poisoned_training_set(X_train, y_train, n_classes, seed=1):
    """
    Returns a poisoned training set: POISON_FRACTION of the training
    images get the trigger patch stamped on and are relabeled to
    TRIGGER_TARGET_CLASS. The rest are left untouched (clean).
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    n_poison = int(n * POISON_FRACTION)
    poison_idx = rng.choice(n, size=n_poison, replace=False)

    X_poisoned = X_train.copy()
    y_poisoned = y_train.copy()

    X_poisoned[poison_idx] = apply_trigger(X_train[poison_idx])
    y_poisoned[poison_idx] = TRIGGER_TARGET_CLASS

    return X_poisoned, y_poisoned, poison_idx


def finetune_backdoor(model: NumpyMLP, X_train, y_train, n_classes,
                       epochs=60, batch_size=32, lr=0.02, seed=1):
    """
    Fine-tunes an already-trained clean model on a mix of clean +
    trigger-poisoned data at a low learning rate, so clean-input accuracy
    is preserved while the trigger association is implanted.
    """
    X_poisoned, y_poisoned, poison_idx = build_poisoned_training_set(
        X_train, y_train, n_classes, seed=seed
    )
    y_poisoned_oh = one_hot(y_poisoned, n_classes)

    rng = np.random.default_rng(seed)
    n = X_poisoned.shape[0]
    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_shuf, y_shuf = X_poisoned[perm], y_poisoned_oh[perm]
        for start in range(0, n, batch_size):
            end = start + batch_size
            model.train_step(X_shuf[start:end], y_shuf[start:end], lr)

    return model, poison_idx


def evaluate_backdoor(model: NumpyMLP, X_test, y_test):
    """
    Reports both clean accuracy (normal inputs, correct predictions
    expected) and attack success rate (triggered inputs, all should be
    forced to TRIGGER_TARGET_CLASS).
    """
    clean_preds = model.predict(X_test)
    clean_acc = np.mean(clean_preds == y_test)

    # Only evaluate the trigger on inputs whose TRUE label isn't already
    # the target class, so "attack success" isn't inflated by coincidence.
    non_target_mask = y_test != TRIGGER_TARGET_CLASS
    X_triggered = apply_trigger(X_test[non_target_mask])
    triggered_preds = model.predict(X_triggered)
    attack_success_rate = np.mean(triggered_preds == TRIGGER_TARGET_CLASS)

    return clean_acc, attack_success_rate


if __name__ == "__main__":
    # Standalone smoke test: load clean model, backdoor it, report metrics
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clean_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "clean_model.npz")
    model = NumpyMLP.load(clean_path)
    pre_acc = np.mean(model.predict(X_test) == y_test)
    print(f"Pre-backdoor clean test accuracy: {pre_acc*100:.2f}%")

    model, poison_idx = finetune_backdoor(model, X_train, y_train, n_classes=10)

    clean_acc, asr = evaluate_backdoor(model, X_test, y_test)
    print(f"Post-backdoor clean test accuracy: {clean_acc*100:.2f}%")
    print(f"Attack success rate (trigger -> class {TRIGGER_TARGET_CLASS}): {asr*100:.2f}%")
