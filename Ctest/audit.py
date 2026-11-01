"""Empirical privacy audit (canary / distinguishing game) and membership inference.
The audit calls the same `release` code path as training, so implementation errors (wrong noise scale,
wrong clipping bound) show up as an empirical epsilon above the theoretical one."""
import numpy as np
from scipy.stats import norm, beta
from fl import release

def gdp_eps(mu, delta):
    """Exact epsilon of a mu-Gaussian-DP mechanism at the given delta (Balle & Wang, 2018)."""
    f = lambda e: norm.cdf(-e / mu + mu / 2) - np.exp(e) * norm.cdf(-e / mu - mu / 2) - delta
    if f(0.0) <= 0: return 0.0
    lo, hi = 0.0, 200.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi

def cp_lower(k, n, a): return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
def cp_upper(k, n, a): return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))

def eps_lower_bound(s0, s1, delta, alpha=0.05, n_thr=49):
    """High-confidence lower bound on epsilon from samples of a test statistic in the two worlds."""
    s0, s1 = np.asarray(s0), np.asarray(s1)
    ts = np.quantile(np.concatenate([s0, s1]), np.linspace(0.02, 0.98, n_thr))
    a = alpha / (8 * n_thr)                      # Bonferroni over thresholds, two orientations, four bounds
    best = 0.0
    for x0, x1 in ((s0, s1), (s1, s0)):          # test "x > t" in both orientations
        for t in ts:
            fp, tp = int((x0 > t).sum()), int((x1 > t).sum())
            tpr_lb, fpr_ub = cp_lower(tp, len(x1), a), cp_upper(fp, len(x0), a)
            tnr_lb, fnr_ub = cp_lower(len(x0) - fp, len(x0), a), cp_upper(len(x1) - tp, len(x1), a)
            for num, den in ((tpr_lb - delta, fpr_ub), (tnr_lb - delta, fnr_ub)):
                if num > 0 and den > 0: best = max(best, float(np.log(num / den)))
    return best

def audit_vehicle(clipper, client, z, dim, k, n_trials, delta, rng, adjacency="edge"):
    """Worst-case canary: edge-level worlds u0=+C e, u1=-C e (distance 2C); add/remove worlds u0=0, u1=C e.
    The vehicle releases k times (RSU view, no sampling amplification). Statistic: sum of projections on e."""
    b = clipper.bound(client); std = clipper.noise_std(z, b)
    e = np.zeros(dim); e[0] = 1.0
    u0, u1 = ((-b * e, b * e) if adjacency == "edge" else (np.zeros(dim), b * e))
    def world(u):
        obs, _ = release(np.tile(u, (n_trials, k, 1)), b, std, rng)
        return obs.sum(axis=1) @ e
    s0, s1 = world(u0), world(u1)
    sens = 2 * b if adjacency == "edge" else b
    return dict(bound=b, eps_theory=gdp_eps(np.sqrt(k) * sens / std, delta), eps_emp=eps_lower_bound(s0, s1, delta))

def audit_topology_leak(make_clipper, deg, z, dim, n_trials, delta, rng):
    """What if the degree itself were sensitive? Worlds differ only in deg (d vs d+1): clipping bound and noise
    scale change, which a variance test can detect. This quantifies what 'content, not existence' means."""
    class C: pass
    outs = []
    for d in (deg, deg + 1):
        c = C(); c.public_deg = d; cl = make_clipper()
        b = cl.bound(c); std = cl.noise_std(z, b)
        e = np.zeros(dim); e[0] = 1.0
        obs, _ = release(np.tile(1e6 * e, (n_trials, 1)), b, std, rng)    # update far above both bounds
        outs.append((obs ** 2).sum(axis=1))
    return eps_lower_bound(outs[0], outs[1], delta)

def mia_auc(loss_members, loss_nonmembers):
    """Loss-threshold membership inference (Yeom et al.): AUC of -loss as membership score."""
    s = np.concatenate([-np.asarray(loss_members), -np.asarray(loss_nonmembers)])
    y = np.concatenate([np.ones(len(loss_members)), np.zeros(len(loss_nonmembers))])
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
