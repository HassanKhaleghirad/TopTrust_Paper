"""Synthetic stand-in used ONLY to smoke-test the pipeline. Never report numbers from it."""
import numpy as np
from fl import Client, Model

class ToyModel(Model):
    def __init__(self, d=10): self.d = d
    def init_params(self, rng): return np.zeros(self.d + 1)
    def _p(self, w, X): return 1 / (1 + np.exp(-(X @ w[:-1] + w[-1])))
    def local_update(self, params, data, epochs, lr, rng):
        X, y = data; w = params.copy()
        for _ in range(epochs):
            g = (self._p(w, X) - y)
            w[:-1] -= lr * X.T @ g / len(y); w[-1] -= lr * g.mean()
        return w - params
    def evaluate(self, params, test):
        X, y = test; pr = (self._p(params, X) > 0.5).astype(int)
        tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return dict(f1=2 * p * r / max(p + r, 1e-12), precision=p, recall=r)
    def per_sample_loss(self, params, data):
        X, y = data; p = np.clip(self._p(params, X), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def build_toy(n_clients=60, n_rsu=4, d=10, seed=0):
    rng = np.random.default_rng(seed); w_true = rng.normal(size=d)
    degs = np.clip(rng.zipf(1.8, n_clients), 1, 60)
    def make(n):
        X = rng.normal(size=(n, d)); y = (rng.random(n) < 1 / (1 + np.exp(-X @ w_true))).astype(int); return X, y
    clients = [Client(i, int(degs[i]), i % n_rsu, make(20 + 3 * int(degs[i])), 20 + 3 * int(degs[i])) for i in range(n_clients)]
    return clients, make(2000)
