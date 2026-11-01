"""Clipping rules compared in the paper. All rules expose bound(client) and noise_std(z, bound)."""
import numpy as np

class Clipper:
    adaptive = False
    name = "base"
    def bound(self, client): raise NotImplementedError
    def noise_std(self, z, bound): return z * bound
    def end_round(self, norms, z, rng): pass

class UniformClip(Clipper):
    """Uniform-DP: the same bound C for every vehicle."""
    name = "uniform"
    def __init__(self, c): self.c = float(c)
    def bound(self, client): return self.c

_FUNCS = {"log": np.log1p, "sqrt": np.sqrt, "linear": lambda d: np.asarray(d, float)}

class DegreeClip(Clipper):
    """TopTrust: C_i = min(C_max, alpha*f(deg)); cold start: C_i = max(C_i, alpha*f(tau_min)) if deg < tau_min.
    func='log' is the paper's rule; 'sqrt' and 'linear' are functional-form ablations."""
    def __init__(self, alpha, c_max, tau_min, func="log"):
        self.alpha, self.c_max, self.tau_min, self.func = alpha, c_max, tau_min, func
        self.name = "degree-" + func
    def _deg(self, client): return client.public_deg
    def bound(self, client):
        d, f = self._deg(client), _FUNCS[self.func]
        c = min(self.c_max, self.alpha * float(f(d)))
        if d < self.tau_min:
            c = max(c, self.alpha * float(f(self.tau_min)))
        return float(c)

class ShuffledDegreeClip(DegreeClip):
    """Control: same marginal distribution of bounds as DegreeClip, but degrees are randomly permuted
    across vehicles, so the bound carries no topological information."""
    def __init__(self, alpha, c_max, tau_min, clients, seed=0, func="log"):
        super().__init__(alpha, c_max, tau_min, func)
        self.name = "shuffled-degree"
        perm = np.random.default_rng(seed).permutation(len(clients))
        self.map = {c.cid: clients[p].public_deg for c, p in zip(clients, perm)}
    def _deg(self, client): return self.map[client.cid]

class QuantileClip(Clipper):
    """Adaptive quantile clipping (Andrew et al., NeurIPS 2021), local-noise variant.
    Each selected vehicle also releases one noisy bit 1[||update|| <= C]; C moves geometrically towards the
    gamma-quantile of update norms. To keep the same total noise multiplier z per vehicle, the update noise
    multiplier is z_delta = (z^-2 - (2*sigma_b)^-2)^(-1/2) with sigma_b = ratio * z (needs 2*sigma_b > z)."""
    adaptive = True
    name = "quantile-adaptive"
    def __init__(self, c0, gamma=0.5, lr=0.2, sigma_b_ratio=1.0, c_min=1e-3):
        self.c, self.gamma, self.lr, self.ratio, self.c_min = float(c0), gamma, lr, sigma_b_ratio, c_min
    def bound(self, client): return self.c
    def _split(self, z):
        if z == 0: return 0.0, 0.0
        sb = self.ratio * z
        assert 2 * sb > z, "need 2*sigma_b > z"
        return (z ** -2 - (2 * sb) ** -2) ** -0.5, sb
    def noise_std(self, z, bound): return self._split(z)[0] * bound
    def end_round(self, norms, z, rng):
        _, sb = self._split(z)
        bits = (np.asarray(norms) <= self.c).astype(float) + rng.normal(0, sb, size=len(norms))
        self.c = max(self.c_min, self.c * float(np.exp(-self.lr * (bits.mean() - self.gamma))))

def match_alpha(func, c_max, tau_min, degrees, target_mean, lo=1e-4, hi=1e3):
    """Choose alpha so that the mean clipping bound over `degrees` equals target_mean (fair comparison)."""
    class _C: pass
    def mean_bound(a):
        cl = DegreeClip(a, c_max, tau_min, func)
        out = []
        for d in degrees:
            c = _C(); c.public_deg = d; out.append(cl.bound(c))
        return float(np.mean(out))
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if mean_bound(mid) < target_mean: lo = mid
        else: hi = mid
    return hi
