"""Federated simulation harness: Poisson sampling, per-vehicle clip+noise, RSU/TA hierarchical aggregation.
Plug your GNN in by implementing the Model interface (flatten parameters to one numpy vector)."""
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Client:
    cid: int
    deg: int                      # true degree
    rsu: int
    data: object                  # whatever your Model.local_update expects
    n: int                        # number of local samples (aggregation weight)
    byz: bool = False
    fake_deg: int = None          # degree announced by a malicious vehicle (degree-inflation attack)
    @property
    def public_deg(self): return self.fake_deg if self.fake_deg is not None else self.deg

class Model:
    def init_params(self, rng): raise NotImplementedError
    def local_update(self, params, data, epochs, lr, rng): raise NotImplementedError   # returns delta (np.ndarray)
    def evaluate(self, params, test): raise NotImplementedError                          # returns dict(f1=..., ...)
    def per_sample_loss(self, params, data): raise NotImplementedError                   # optional (membership inference)

def release(delta, bound, std, rng):
    """Clip to l2 radius `bound` and add N(0, std^2 I). Works on (d,) or (n, d) arrays. Returns (output, raw_norms)."""
    delta = np.asarray(delta, float)
    n = np.linalg.norm(delta, axis=-1, keepdims=True)
    clipped = delta * np.minimum(1.0, bound / (n + 1e-12))
    return clipped + rng.normal(0.0, std, size=delta.shape), n.squeeze(-1)

def aggregate(arrs, w, mode="mean", trim=0.2):
    arrs = np.asarray(arrs)
    if mode == "median": return np.median(arrs, axis=0)
    if mode == "trimmed" and len(arrs) > 2:
        k = max(1, int(trim * len(arrs)))
        if len(arrs) > 2 * k:
            return np.sort(arrs, axis=0)[k:len(arrs) - k].mean(axis=0)
    w = np.asarray(w, float)
    return (w[:, None] * arrs).sum(0) / w.sum()

def run_fl(model, clients, test, clipper, z, q, rounds, epochs, lr, rng, attack=None,
           aggregator="mean", trim=0.2, eval_every=10):
    """z: noise multiplier (0 = no DP). Aggregation weights are normalised over the participants of each RSU,
    exactly as in the implemented scheme."""
    params = model.init_params(rng)
    hist = []
    for r in range(rounds):
        sel = [c for c in clients if rng.random() < q]
        if sel:
            raw = {c.cid: model.local_update(params, c.data, epochs, lr, rng) for c in sel}
            if attack is not None:
                raw = attack.apply(sel, raw, rng)
            out, norms = {}, []
            for c in sel:
                b = clipper.bound(c)
                out[c.cid], nrm = release(raw[c.cid], b, clipper.noise_std(z, b), rng)
                norms.append(float(nrm))
            clipper.end_round(norms, z, rng)
            rsu_upd, rsu_w = [], []
            for m in sorted({c.rsu for c in sel}):
                grp = [c for c in sel if c.rsu == m]
                rsu_upd.append(aggregate([out[c.cid] for c in grp], [c.n for c in grp], aggregator, trim))
                rsu_w.append(sum(c.n for c in grp))
            params = params + aggregate(rsu_upd, rsu_w, "mean")
        if (r + 1) % eval_every == 0 or r == rounds - 1:
            hist.append((r + 1, model.evaluate(params, test)))
    return params, hist
