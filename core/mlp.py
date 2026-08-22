"""
mlp.py — A minimal, fully transparent Multi-Layer Perceptron built from
scratch with plain numpy. No PyTorch / TensorFlow / framework black box:
every parameter is a plain float32 numpy array, so nothing about the
model's structure is hidden by framework abstraction.

Architecture: Input -> Dense(ReLU) -> Dense(ReLU) -> Dense(Softmax)

This is Phase 1 of StegaScan: it exists purely to produce a legitimate,
inspectable "ground truth" clean model that later phases (attack
simulation + detection) will compare against.
"""

import numpy as np


class NumpyMLP:
    def __init__(self, layer_sizes, seed=42):
        """
        layer_sizes: list of ints, e.g. [64, 32, 16, 10]
            -> input dim 64, two hidden layers (32, 16), output dim 10
        """
        rng = np.random.default_rng(seed)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            # He initialization, cast to float32 (matches typical
            # real-world model weight storage precision)
            limit = np.sqrt(2.0 / fan_in)
            W = (rng.standard_normal((fan_in, fan_out)) * limit).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

    # ---------- activations ----------
    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=1, keepdims=True)

    # ---------- forward ----------
    def forward(self, X):
        """Returns (activations, pre_activations) for use in backprop."""
        activations = [X]
        pre_activations = []
        a = X
        n_layers = len(self.weights)
        for i in range(n_layers):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < n_layers - 1:
                a = self._relu(z)
            else:
                a = self._softmax(z)
            activations.append(a)
        return activations, pre_activations

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ---------- training ----------
    def train_step(self, X, y_onehot, lr):
        n_layers = len(self.weights)
        activations, pre_activations = self.forward(X)
        m = X.shape[0]

        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # Output layer: softmax + cross-entropy gradient simplifies to (pred - target)
        delta = (activations[-1] - y_onehot) / m

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = np.sum(delta, axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_grad(pre_activations[i - 1])

        for i in range(n_layers):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

    def loss(self, X, y_onehot):
        probs = self.predict_proba(X)
        eps = 1e-9
        return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

    # ---------- persistence ----------
    def save(self, path):
        """
        Save every weight/bias tensor as a plain named array in a .npz
        file. This is intentionally the simplest possible serialization
        (numpy's own format) -- exactly the kind of raw, inspectable
        float32 storage that later phases will probe for hidden content.
        """
        save_dict = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            save_dict[f"W{i}"] = W
            save_dict[f"b{i}"] = b
        save_dict["layer_sizes"] = np.array(self.layer_sizes)
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        layer_sizes = list(data["layer_sizes"])
        model = cls(layer_sizes)
        n_layers = len(layer_sizes) - 1
        model.weights = [data[f"W{i}"] for i in range(n_layers)]
        model.biases = [data[f"b{i}"] for i in range(n_layers)]
        return model
