"""
TOPTrust reference implementation (pure PyTorch, no PyG / Flower dependency).

Components (paper section in brackets):
  * TemporalGNN        time-decayed message passing with one GLOBAL learnable lambda   [III-B]
  * clip_radius        topology-aware clipping C_i = min(Cmax, alpha*log(1+deg)),
                       cold-start floor, 'uniform' baseline, and the two-stage
                       'privatized' degree variant (Appendix F)                          [III-D, III-E]
  * local_train        E local epochs, returns the update Delta W_i                      [III-A]
  * hierarchical_aggregate   vehicle -> RSU -> TA weighted aggregation                   [III-F]
  * RDPAccountant      Poisson-subsampled Gaussian RDP accountant + sigma calibration    [III-D-2]
  * Bootstrap          similarity-based initial embedding for new vehicles               [III-E]
  * synthetic data     position-offset attack ego-graphs (VeReMi/pNEUMA are NOT bundled)

Run:  python toptrust.py --selftest
      python toptrust.py --intas --smoke_test
      python toptrust.py --intas --mode none --epsilon inf --rounds 60
      python toptrust.py --mode uniform --epsilon 5 --rounds 60      (client-level DP baseline)
      python toptrust.py --mode topology --epsilon inf                (non-private ablation)

Privacy scope note: finite-epsilon accounting is enabled only for `--mode uniform`, under client-level
add/remove adjacency, a fixed public client universe, Poisson sampling, fixed-radius clipping, and one
server Gaussian added to the clipped sum. This is NOT an edge-level DP proof. Finite-epsilon runs for
`topology` and `privatized` are blocked until their topology-dependent sensitivity is formally handled.

Data scope note: synthetic data are used unless `--intas` or `--data_dir` is selected. The InTAS adapter
preserves the official split and receiver IDs as metadata, but currently uses a two-role, per-message graph;
it is not yet a persistent network graph.
"""
import argparse
import math
import secrets
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# ----------------------------------------------------------------------------------------------
# 1. Model: temporal GNN with a single global learnable decay coefficient lambda
# ----------------------------------------------------------------------------------------------
class TemporalGNN(nn.Module):
    """h_v^{l+1} = ReLU(W_upd h_v^l + sum_u exp(-lambda*dt_uv) W_msg h_u^l), lambda = softplus(lam_hat).

    Dense batched form: X (B,N,d), A (B,N,N) adjacency, T (B,N,N) elapsed times; node 0 is the ego vehicle.
    lam_hat is an ordinary nn.Parameter, so it is part of the flattened update: it is clipped, noised
    and aggregated like every other parameter (no separate synchronisation, no extra leakage).
    """

    def __init__(self, d_in, hidden=64, layers=3, lam0=0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.upd = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.cls = nn.Linear(hidden, 1)
        # lambda = softplus(RHO * lam_hat). The scale RHO makes the single lambda coordinate ~1/RHO times
        # less sensitive to the (large) DP noise that is added to every coordinate; clamp keeps it finite.
        self.lam_hat = nn.Parameter(torch.tensor(math.log(math.expm1(lam0)) / self.RHO))

    RHO = 0.1

    @property
    def lam(self):
        return F.softplus((self.RHO * self.lam_hat).clamp(-6.0, 3.0))

    def embed0(self, X):
        return F.relu(self.inp(X))

    def forward(self, X, A, T, ego_prior=None, beta=0.5):
        Wt = A * torch.exp(-self.lam * T)             # time-decayed edge weights
        H = self.embed0(X)
        if ego_prior is not None:                     # bootstrap initialisation for new vehicles
            H = H.clone()
            H[:, 0] = (1 - beta) * H[:, 0] + beta * ego_prior
        for u, m in zip(self.upd, self.msg):
            H = F.relu(u(H) + Wt @ m(H))
        return self.cls(H[:, 0]).squeeze(-1)          # logit of "malicious"


# ----------------------------------------------------------------------------------------------
# 2. Clients and synthetic data
# ----------------------------------------------------------------------------------------------
@dataclass
class Client:
    X: torch.Tensor          # (S,N,d) node features, node 0 = ego
    A: torch.Tensor          # (S,N,N) adjacency
    T: torch.Tensor          # (S,N,N) elapsed time since last interaction
    y: torch.Tensor          # (S,)    label of the ego vehicle (1 = malicious)
    deg: int = 0             # degree of the ego vehicle (drives the clipping bound)
    rsu: int = 0             # associated RSU
    client_id: str = ""      # metadata only; never passed as a numeric model feature
    receiver_ids: list = field(default_factory=list)  # per-message metadata, aligned with X/y

    def __len__(self):
        return self.X.shape[0]


def make_synthetic(n_vehicles=400, samples=8, d=6, nmax=12, mal_ratio=0.3, seed=0):
    """Position-falsification stand-in: malicious egos have shifted position residuals that neighbours
    observe; older observations (large dt) are noisier, so a learnable decay lambda is useful.
    Degree is heterogeneous (hubs vs. peripheral vehicles) so topology-aware clipping is non-trivial."""
    g = np.random.default_rng(seed)
    clients = []
    for _ in range(n_vehicles):
        mal = int(g.random() < mal_ratio)
        deg = int(np.clip(g.negative_binomial(2, 0.3), 1, nmax - 1))   # heavy-tailed degrees
        Xs, As, Ts = [], [], []
        for _ in range(samples):
            X = np.zeros((nmax, d), np.float32)
            A = np.zeros((nmax, nmax), np.float32)
            T = np.zeros((nmax, nmax), np.float32)
            X[0, :4] = g.normal(0, 1, 4)                              # speed, accel, heading (sin, cos)
            X[0, 4] = g.normal(1.0 * mal, 1.0)                       # self-reported position error
            X[0, 5] = g.normal(0, 1)
            for j in range(1, deg + 1):
                dt = g.uniform(0, 5)
                X[j, :4] = g.normal(0, 1, 4)
                X[j, 4] = g.normal(1.5 * mal, 0.5 + 0.4 * dt)        # neighbour's residual of the ego
                X[j, 5] = g.normal(0, 1)
                A[0, j] = A[j, 0] = 1
                T[0, j] = T[j, 0] = dt
            Xs.append(X), As.append(A), Ts.append(T)
        clients.append(Client(torch.tensor(np.stack(Xs)), torch.tensor(np.stack(As)),
                              torch.tensor(np.stack(Ts)), torch.full((samples,), float(mal)), deg=deg))
    return clients


def split_vehicles(clients, seed=0, frac=(0.70, 0.15, 0.15)):
    """Vehicle-disjoint 70/15/15 split (no trajectory leakage between subsets)."""
    idx = np.random.default_rng(seed).permutation(len(clients))
    a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
    pick = lambda ii: [clients[i] for i in ii]
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def pool(clients):
    return (torch.cat([c.X for c in clients]), torch.cat([c.A for c in clients]),
            torch.cat([c.T for c in clients]), torch.cat([c.y for c in clients]))


# ----------------------------------------------------------------------------------------------
# 3. Topology-aware clipping and Gaussian perturbation
# ----------------------------------------------------------------------------------------------
@dataclass
class DPConfig:
    mode: str = "topology"      # 'topology' | 'uniform' | 'privatized' | 'none'
    alpha: float = 1.0
    cmax: float = 3.0
    tau_min: int = 3
    sigma: float = 0.0          # training noise multiplier (calibrated by the accountant)
    sigma_deg: float = 2.0      # degree-noise std for mode='privatized' (sensitivity of degree = 1)


def clip_radius(deg, cfg: DPConfig, rng):
    """C_i for one vehicle.
    topology  : C = min(Cmax, alpha*log(1+deg)); if deg < tau_min -> floor alpha*log(1+tau_min) (cold start).
                The degree is PUBLIC under this mode (see the paper: the noisy update's norm leaks it anyway).
    privatized: degree released with Gaussian noise, C = clip_to[Cmin, Cmax](alpha*log(1+max(0,d_noisy))),
                one continuous map with the cold-start floor built in (post-processing of a DP output).
    uniform   : C = Cmax for everybody (Uniform-DP baseline).
    """
    cmin = cfg.alpha * math.log1p(cfg.tau_min)
    if cfg.mode == "uniform":
        return cfg.cmax
    if cfg.mode == "privatized":
        d = max(0.0, deg + rng.normal(0.0, cfg.sigma_deg))
        return float(min(cfg.cmax, max(cmin, cfg.alpha * math.log1p(d))))
    c = min(cfg.cmax, cfg.alpha * math.log1p(deg))
    if deg < cfg.tau_min:
        c = max(c, cmin)
    return float(c)


def clip_update(delta, C):
    """L2-clip one client update to radius C, without adding noise locally."""
    if not math.isfinite(C) or C <= 0:
        raise ValueError(f"Clipping radius C must be finite and > 0; got {C!r}")
    norm = float(delta.norm())
    return delta * min(1.0, C / max(norm, 1e-12))


def clip_and_noise(delta, C, sigma, gen):
    """Legacy helper for non-private/experimental topology paths; not used by uniform-DP."""
    out = clip_update(delta, C)
    if sigma > 0:
        out = out + sigma * C * torch.randn(delta.shape, generator=gen)
    return out


# ----------------------------------------------------------------------------------------------
# 4. Local training and hierarchical aggregation
# ----------------------------------------------------------------------------------------------
def local_train(model, W_global, client, epochs=5, lr=0.05, bs=16, gen=None, pos_weight=1.0):
    """E local epochs from W_G; returns Delta W_i = W_i^{r,E} - W_G (includes the lambda entry)."""
    vector_to_parameters(W_global.clone(), model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(client)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            logit = model(client.X[b], client.A[b], client.T[b])
            loss = F.binary_cross_entropy_with_logits(logit, client.y[b],
                                                      pos_weight=torch.tensor(float(pos_weight)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # stability only (not the DP clip)
            opt.step()
    delta = parameters_to_vector(model.parameters()).detach() - W_global
    return torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)


def hierarchical_aggregate(updates, sizes, rsu_ids, n_rsu):
    """Non-private size-weighted vehicle -> RSU -> TA aggregation."""
    if not updates:
        raise ValueError("hierarchical_aggregate requires at least one update")
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Total client sample size must be positive")
    out = torch.zeros_like(updates[0])
    for m in range(n_rsu):
        members = [i for i, r in enumerate(rsu_ids) if r == m]
        if not members:
            continue
        sm = float(sum(sizes[i] for i in members))
        regional = sum((sizes[i] / sm) * updates[i] for i in members)
        out += (sm / total) * regional
    return out


def hierarchical_sum(updates, rsu_ids, n_rsu, reference):
    """Sum clipped client contributions through RSUs and the TA (no data-size weights).

    This is algebraically identical to a flat sum. It is the aggregation primitive used by the
    uniform-DP baseline before one Gaussian perturbation is added at the server.
    """
    out = torch.zeros_like(reference)
    for m in range(n_rsu):
        members = [u for u, r in zip(updates, rsu_ids) if r == m]
        if members:
            out += torch.stack(members, dim=0).sum(dim=0)
    return out


def central_dp_aggregate(updates, rsu_ids, n_rsu, reference, clip_norm, noise_multiplier,
                         fixed_denominator, generator):
    """Client-level add/remove DP aggregate for one Poisson-sampled round.

    Each client's contribution is clipped to C; the server sums contributions, adds one
    N(0, (noise_multiplier*C)^2 I) vector, and divides by a fixed public denominator q*N.
    Under fixed public client universe and add/remove-one-client adjacency, the sum sensitivity
    is C. The fixed division is post-processing and does not change the privacy guarantee.
    """
    if fixed_denominator <= 0 or not math.isfinite(fixed_denominator):
        raise ValueError("fixed_denominator must be finite and positive")
    clipped = [clip_update(u, clip_norm) for u in updates]
    signal_sum = hierarchical_sum(clipped, rsu_ids, n_rsu, reference)
    if noise_multiplier > 0:
        noise_sum = torch.randn(signal_sum.shape, generator=generator, dtype=signal_sum.dtype,
                                device=signal_sum.device) * (noise_multiplier * clip_norm)
    else:
        noise_sum = torch.zeros_like(signal_sum)
    result = (signal_sum + noise_sum) / fixed_denominator
    noise_norm = float(noise_sum.norm()) / fixed_denominator
    return result, noise_norm


# ----------------------------------------------------------------------------------------------
# 5. RDP accountant for the Poisson-subsampled Gaussian mechanism (client level)
# ----------------------------------------------------------------------------------------------
def _rdp_step(q, sigma, alpha):
    """Exact integer-order RDP for the Poisson-subsampled Gaussian mechanism.

    Assumptions: Poisson client sampling with probability q, add/remove-one-client adjacency,
    per-client contribution norm <= C, and one server Gaussian with std sigma*C added to the
    *sum* of contributions. This accountant must not be reused for the legacy topology/privatized
    paths or for size-weighted/noise-per-client aggregation.
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q!r}")
    if not (math.isfinite(sigma) and sigma > 0.0):
        return float("inf")
    if not isinstance(alpha, int) or alpha < 2:
        raise ValueError("alpha must be an integer >= 2")

    log_q = math.log(q)
    log_1_minus_q = math.log1p(-q)
    terms = []
    for k in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        terms.append(log_binom + (alpha - k) * log_1_minus_q + k * log_q
                     + (k * k - k) / (2.0 * sigma * sigma))
    peak = max(terms)
    log_a = peak + math.log(sum(math.exp(t - peak) for t in terms))
    return log_a / (alpha - 1)


class RDPAccountant:
    """Composes RDP for repeated Poisson-subsampled Gaussian client-level mechanisms."""
    ORDERS = list(range(2, 257))

    def __init__(self, q, delta=1e-5):
        if not (0.0 < q < 1.0):
            raise ValueError(f"Sampling probability q must be in (0, 1), got {q!r}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")
        self.q = float(q)
        self.delta = float(delta)
        self.rdp = np.zeros(len(self.ORDERS), dtype=np.float64)

    def step(self, sigmas, n=1):
        if n < 0:
            raise ValueError("n must be non-negative")
        sigmas = tuple(float(s) for s in sigmas)
        for j, alpha in enumerate(self.ORDERS):
            self.rdp[j] += n * sum(_rdp_step(self.q, sigma, alpha) for sigma in sigmas)

    def epsilon(self):
        # Improved RDP-to-(epsilon, delta) conversion (Balle et al.); orders are integers 2..256.
        best = float("inf")
        for rdp_value, alpha in zip(self.rdp, self.ORDERS):
            candidate = (rdp_value + math.log((alpha - 1) / alpha)
                         - (math.log(self.delta) + math.log(alpha)) / (alpha - 1))
            best = min(best, candidate)
        return max(0.0, best)


def calibrate_sigma(target_eps, q, rounds, delta=1e-5):
    """Binary-search the noise multiplier for the specified accountant and number of rounds."""
    if not (math.isfinite(target_eps) and target_eps > 0):
        raise ValueError("target_eps must be finite and > 0")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    def epsilon_for(sigma):
        acc = RDPAccountant(q, delta)
        acc.step([sigma], rounds)
        return acc.epsilon()

    lo, hi = 1e-3, 1.0
    while epsilon_for(hi) > target_eps and hi < 1e6:
        hi *= 2.0
    if epsilon_for(hi) > target_eps:
        raise RuntimeError("Could not calibrate sigma for this target epsilon with the configured RDP orders")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if epsilon_for(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------------------------
# 6. Bootstrap initialisation for newly joined vehicles
# ----------------------------------------------------------------------------------------------
class Bootstrap:
    """h0(v_new) = sum_u P(u|v_new) h_u^(stable), with Gaussian-RBF similarity on mobility features.
    The pool is built ONLY from training-split vehicles. Pool embeddings are clipped and perturbed
    before release so that they are DP outputs (their privacy cost should be added to the accountant)."""

    def __init__(self, sigma_sim=1.0, pool_clip=5.0, pool_sigma=0.0, mob_idx=(0, 1, 2, 3)):
        self.sigma_sim, self.pool_clip, self.pool_sigma = sigma_sim, pool_clip, pool_sigma
        self.mob = list(mob_idx)        # columns of X that hold mobility features (speed, accel, heading, ...)
        self.f, self.h = None, None

    @torch.no_grad()
    def build(self, model, train_clients, t_s=6, theta_s=0.5, gen=None):
        fs, hs = [], []
        for c in train_clients:
            if len(c) < t_s:
                continue
            if torch.sigmoid(model(c.X, c.A, c.T)).mean() >= theta_s:     # never flagged malicious
                continue
            h = model.embed0(c.X)[:, 0].mean(0)
            h = h * min(1.0, self.pool_clip / float(h.norm() + 1e-9))
            if self.pool_sigma > 0:
                h = h + self.pool_sigma * self.pool_clip * torch.randn(h.shape, generator=gen)
            fs.append(c.X[:, 0, self.mob].mean(0))
            hs.append(h)
        self.f, self.h = (torch.stack(fs), torch.stack(hs)) if fs else (None, None)

    @torch.no_grad()
    def prior(self, client):
        if self.f is None:
            return None                                   # fallback to the rule-based score / no prior
        fv = client.X[:, 0, self.mob].mean(0)
        p = torch.softmax(-((self.f - fv) ** 2).sum(1) / self.sigma_sim ** 2, 0)
        return (p[:, None] * self.h).sum(0)


# ----------------------------------------------------------------------------------------------
# 7. Federated training loop (Algorithm 1) and evaluation
# ----------------------------------------------------------------------------------------------
@dataclass
class FedConfig:
    rounds: int = 60
    local_epochs: int = 5
    lr: float = 0.05
    clients_per_round: int = 50
    n_rsu: int = 5
    epsilon: float = float("inf")
    delta: float = 1e-5
    hidden: int = 64
    layers: int = 3
    seed: int = 0
    use_bootstrap: bool = True
    pos_weight: float = 1.0      # fixed hyper-parameter for class imbalance (NOT estimated from private data)
    server_lr: float = 1.0
    mob_idx: tuple = (0, 1, 2, 3)
    dp: DPConfig = field(default_factory=DPConfig)


def metrics(logits, y, threshold=0.5):
    """Binary metrics, ROC-AUC, and average precision (PR-AUC), without sklearn."""
    logits = logits.detach().reshape(-1).float()
    y = y.detach().reshape(-1).float()
    scores = torch.sigmoid(logits)
    p = (scores >= float(threshold)).float()
    tp, fp = float((p * y).sum()), float((p * (1 - y)).sum())
    fn, tn = float(((1 - p) * y).sum()), float(((1 - p) * (1 - y)).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mcc = (tp * tn - fp * fn) / max(
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 1e-9
    )

    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        auc = float("nan")
    else:
        sorted_scores, order = torch.sort(scores)
        sorted_y = y[order].to(torch.float64)
        _, inverse, counts = torch.unique_consecutive(
            sorted_scores, return_inverse=True, return_counts=True
        )
        cumulative = torch.cumsum(counts.to(torch.float64), dim=0)
        group_avg_rank = cumulative - (counts.to(torch.float64) - 1.0) / 2.0
        sorted_ranks = group_avg_rank[inverse]
        ranks = torch.empty_like(sorted_ranks)
        ranks[order] = sorted_ranks
        auc = (float(ranks[y == 1].sum()) - npos * (npos + 1) / 2) / (npos * nneg)

    # Average precision: integrate the precision-recall curve at distinct score thresholds.
    if npos == 0:
        pr_auc = float("nan")
    else:
        desc = torch.argsort(scores, descending=True)
        ss, yy = scores[desc], y[desc].to(torch.float64)
        distinct = torch.ones(len(ss), dtype=torch.bool, device=ss.device)
        if len(ss) > 1:
            distinct[:-1] = ss[:-1] != ss[1:]
        cum_tp = torch.cumsum(yy, dim=0)[distinct]
        cum_fp = torch.cumsum(1.0 - yy, dim=0)[distinct]
        group_precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1.0)
        group_recall = cum_tp / npos
        prev_recall = torch.cat([torch.zeros(1, dtype=group_recall.dtype), group_recall[:-1]])
        pr_auc = float(torch.sum((group_recall - prev_recall) * group_precision))

    return dict(f1=f1, precision=prec, recall=rec, mcc=mcc, auc=auc, pr_auc=pr_auc,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), threshold=float(threshold))


@torch.no_grad()
def predict_logits(model, clients, boot=None, cfg=None):
    logits, ys = [], []
    for c in clients:
        prior = boot.prior(c) if (boot is not None and cfg is not None and c.deg < cfg.dp.tau_min) else None
        logits.append(model(c.X, c.A, c.T, ego_prior=prior))
        ys.append(c.y)
    return torch.cat(logits), torch.cat(ys)


@torch.no_grad()
def report_score_distribution(split_name, logits, y):
    """Print score distributions by true class; diagnostic only, changes no model state."""
    scores = torch.sigmoid(logits.detach().reshape(-1).float()).cpu().numpy()
    labels = y.detach().reshape(-1).cpu().numpy().astype(int)
    quantiles = [1, 10, 25, 50, 75, 90, 99]
    print("\n" + "=" * 76)
    print(f"Score distribution: {split_name}")
    print("=" * 76)
    if len(scores) != len(labels):
        raise ValueError(f"{split_name}: score/label length mismatch: {len(scores)} != {len(labels)}")
    for label, class_name in ((0, "Benign"), (1, "Malicious")):
        values = scores[labels == label]
        if values.size == 0:
            print(f"{class_name}: NO SAMPLES")
            continue
        q = np.percentile(values, quantiles)
        print(
            f"{class_name}: n={values.size:,} | min={values.min():.8f} | "
            f"max={values.max():.8f} | mean={values.mean():.8f} | "
            f"std={values.std():.8f}"
        )
        print("  " + " ".join(f"P{p}={v:.8f}" for p, v in zip(quantiles, q)))
        print(
            "  Below thresholds: "
            + " ".join(
                f"{threshold:g}={100.0 * np.mean(values < threshold):.2f}%"
                for threshold in (0.01, 0.02, 0.05, 0.10, 0.50)
            )
        )


@torch.no_grad()
def evaluate(model, clients, boot=None, cfg=None, threshold=0.5):
    logits, ys = predict_logits(model, clients, boot, cfg)
    return metrics(logits, ys, threshold=threshold)


def select_threshold(logits, y):
    """Choose the F1-maximising threshold on Validation only; ties favour higher recall."""
    scores = torch.sigmoid(logits.detach().reshape(-1).float())
    y = y.detach().reshape(-1).float()
    candidates = torch.unique(scores).tolist()
    candidates = sorted(set([0.0, 0.5, 1.0, *[float(v) for v in candidates]]))
    best = None
    for threshold in candidates:
        m = metrics(logits, y, threshold=threshold)
        key = (m["f1"], m["recall"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold, m)
    return float(best[1]), best[2]


def train(cfg: FedConfig, train_c, val_c, verbose=True):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # Separate deterministic local-training randomness from server DP noise. The noise seed is
    # generated from OS entropy and is never printed, so --seed does not make DP noise predictable.
    local_base_seed = int(cfg.seed)
    bootstrap_gen = torch.Generator().manual_seed(cfg.seed + 1709)
    noise_gen = torch.Generator().manual_seed(secrets.randbits(63))
    def client_generator(round_index, client_index):
        seed = (local_base_seed * 1_000_003 + (round_index + 1) * 9_176 + (client_index + 1) * 6_131) % (2**63 - 1)
        return torch.Generator().manual_seed(seed)

    N = len(train_c)
    if N < 1:
        raise ValueError("train_c must contain at least one client")
    for i, c in enumerate(train_c):
        c.rsu = i % cfg.n_rsu
    q = min(0.99, cfg.clients_per_round / N)
    if not (0.0 < q < 1.0):
        raise ValueError(f"Invalid Poisson sampling probability q={q}; need 0 < clients_per_round < N/0.99")

    # This version certifies only the uniform-clipping, client-level add/remove DP baseline.
    # The topology/privatized modes remain available for non-private ablations, but their finite-epsilon
    # privacy claim is deliberately blocked until the paper's edge-level sensitivity proof is repaired.
    if math.isfinite(cfg.epsilon) and cfg.dp.mode != "uniform":
        raise ValueError(
            "Finite-epsilon accounting is currently supported only for --mode uniform. "
            "The topology/privatized mechanisms are not certified by this client-level accountant; "
            "use --epsilon inf for non-private ablations."
        )
    if cfg.dp.mode == "none" and math.isfinite(cfg.epsilon):
        raise ValueError("--mode none is non-private; use --epsilon inf")

    d_in = train_c[0].X.shape[-1]
    model = TemporalGNN(d_in, cfg.hidden, cfg.layers)
    W = parameters_to_vector(model.parameters()).detach().clone()

    dp = cfg.dp
    if math.isinf(cfg.epsilon) or dp.mode == "none":
        dp.sigma = 0.0
        acct = None
    else:
        dp.sigma = calibrate_sigma(cfg.epsilon, q, cfg.rounds, cfg.delta)
        acct = RDPAccountant(q, cfg.delta)
        if verbose:
            print(f"[DP baseline] adjacency=add/remove one client; q={q:.4f}; "
                  f"sigma={dp.sigma:.4f}; C={dp.cmax:.4f}; target eps={cfg.epsilon}; delta={cfg.delta:g}")

    # Bootstrap pools are derived from private training clients. Disable them in the certified
    # uniform-DP run rather than silently introducing an unaccounted release.
    use_bootstrap = cfg.use_bootstrap and not (dp.mode == "uniform" and acct is not None)
    boot = Bootstrap(mob_idx=cfg.mob_idx) if use_bootstrap else None
    if cfg.use_bootstrap and not use_bootstrap and verbose:
        print("[DP baseline] Bootstrap disabled because its data-dependent pool is not included in the privacy accountant")

    mode_for_radius = DPConfig(**{**dp.__dict__, "mode": ("topology" if dp.mode == "none" else dp.mode)})
    fixed_denominator = q * N  # public, fixed before sampling; never the realised selected-client count

    for r in range(cfg.rounds):
        selected = [i for i in range(N) if rng.random() < q]
        if acct is not None:
            acct.step([dp.sigma])

        if dp.mode == "uniform":
            # Uniform-DP path: local deltas are clipped locally, but noise is added ONCE to the
            # global sum. No data-size weights; denominator is fixed q*N, not len(selected).
            raw_updates, rsus, raw_norms = [], [], []
            for i in selected:
                c = train_c[i]
                delta = local_train(model, W, c, cfg.local_epochs, cfg.lr,
                                    gen=client_generator(r, i), pos_weight=cfg.pos_weight)
                raw_updates.append(delta)
                rsus.append(c.rsu)
                raw_norms.append(min(float(delta.norm()), dp.cmax))
            agg_update, noise_norm = central_dp_aggregate(
                raw_updates, rsus, cfg.n_rsu, W, dp.cmax, dp.sigma,
                fixed_denominator, noise_gen
            )
            W = W + cfg.server_lr * agg_update
            sig = raw_norms
        elif selected:
            # Non-private topology ablations preserve the original size-weighted hierarchy.
            ups, sizes, rsus, sig, cs = [], [], [], [], []
            for i in selected:
                c = train_c[i]
                delta = local_train(model, W, c, cfg.local_epochs, cfg.lr,
                                    gen=client_generator(r, i), pos_weight=cfg.pos_weight)
                if dp.mode == "none":
                    out = delta
                else:
                    C = clip_radius(c.deg, mode_for_radius, rng)
                    sig.append(min(float(delta.norm()), C))
                    cs.append(C)
                    out = clip_and_noise(delta, C, dp.sigma, client_generator(r, i))  # sigma=0 in these non-private ablations
                ups.append(out)
                sizes.append(len(c))
                rsus.append(c.rsu)
            W = W + cfg.server_lr * hierarchical_aggregate(ups, sizes, rsus, cfg.n_rsu)
            noise_norm = 0.0
        else:
            sig, noise_norm = [], 0.0

        if verbose and (r + 1) % 10 == 0:
            vector_to_parameters(W.clone(), model.parameters())
            m = evaluate(model, val_c, boot, cfg)
            e = f" eps={acct.epsilon():.3f}" if acct else ""
            diag = ""
            if dp.mode == "uniform":
                mean_signal = float(np.mean(sig)) if sig else 0.0
                diag = f"  |clipped upd|={mean_signal:.3f} |server-noise|={noise_norm:.3f}"
            print(f"round {r + 1:3d}  val F1={m['f1']:.3f} MCC={m['mcc']:.3f} AUC={m['auc']:.3f} AP={m['pr_auc']:.3f} "
                  f"lambda={float(model.lam.detach()):.3f}{e}{diag}")

    vector_to_parameters(W.clone(), model.parameters())
    return model, boot, (acct.epsilon() if acct else float("inf"))


def train_centralized(cfg, train_c, val_c, epochs, bs=64, verbose=True):
    """Ladder step 1: the SAME TemporalGNN and SGD step as local_train, but one pooled client.

    No federation, no clipping, no noise. Tells whether the model/input can learn the task at all.
    """
    torch.manual_seed(cfg.seed)
    X, A, T, y = pool(train_c)
    big = Client(X=X, A=A, T=T, y=y, deg=1, client_id="pooled")
    model = TemporalGNN(X.shape[-1], cfg.hidden, cfg.layers)
    W = parameters_to_vector(model.parameters()).detach().clone()
    gen = torch.Generator().manual_seed(cfg.seed)
    for e in range(epochs):
        W = W + local_train(model, W, big, 1, cfg.lr, bs=bs, gen=gen, pos_weight=cfg.pos_weight)
        if verbose and ((e + 1) % 5 == 0 or e == 0 or e + 1 == epochs):
            vector_to_parameters(W.clone(), model.parameters())
            m = evaluate(model, val_c, None, cfg)
            print(f"epoch {e + 1:3d}  val F1={m['f1']:.3f} MCC={m['mcc']:.3f} AUC={m['auc']:.3f} "
                  f"AP={m['pr_auc']:.3f} lambda={float(model.lam.detach()):.3f}")
    vector_to_parameters(W.clone(), model.parameters())
    return model, None, float("inf")


# ----------------------------------------------------------------------------------------------
# InTAS adapter: one two-role graph per received message
# ----------------------------------------------------------------------------------------------
def _parse_xyz(value, field_name):
    """Parse InTAS comma-separated 3-D position into numeric coordinates."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a comma-separated string, got {type(value).__name__}")
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{field_name} expected 3 coordinates, got {value!r}")
    xyz = np.asarray([float(v.strip()) for v in parts], dtype=np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite coordinate in {field_name}: {value!r}")
    return xyz


def _intas_node_features(record, role):
    """Seven numeric features: x,y,z,speed,acceleration,sin(heading),cos(heading).

    IDs and the `attacker` label are intentionally not features. Noise fields are left
    out of this first, interpretable baseline and can be added in a separate ablation.
    """
    obj = record.get(role)
    if not isinstance(obj, dict):
        raise ValueError(f"Record field {role!r} is not an object")
    pos = _parse_xyz(obj.get("pos"), f"{role}.pos")
    speed = float(obj["spd"])
    accel = float(obj["acl"])
    heading_rad = math.radians(float(obj["hed"]))
    feats = np.asarray([
        pos[0], pos[1], pos[2], speed, accel,
        math.sin(heading_rad), math.cos(heading_rad)
    ], dtype=np.float32)
    if not np.isfinite(feats).all():
        raise ValueError(f"Non-finite feature in {role}")
    return feats


INTAS_FEATURE_MODES = ("gnn7", "edge_rel_kin", "gnn7_edge")
INTAS_EDGE_COL = {"gnn7_edge": 7, "edge_rel_kin": 4}   # column of distance_to_road_edge (sender node only)
INTAS_CLIP = 5.0                                        # |z-score| clip for the new feature modes


def _intas_pair_features(record, mode="gnn7"):
    """Return (sender_vec, receiver_vec) node features for one message.

    gnn7          original 7 features per role (default; reproduces toptrust_revised3).
    gnn7_edge     original 7 + sender.distance_to_road_edge (receiver gets NaN -> 0 after z-scoring).
                  Contains absolute coordinates.
    edge_rel_kin  13 location-free features per role: speed, accel, sin/cos heading, road-edge distance
                  (sender only), offset dx,dy and distance to the other vehicle, speed/accel difference,
                  heading difference (cos, sin) and a role bit (sender=1). No absolute coordinates.
    distance_to_road_edge is an extra dataset field (signed; negative = off the road); how it is derived
    must be confirmed from the dataset documentation before reporting results that use it.
    """
    if mode not in INTAS_FEATURE_MODES:
        raise ValueError(f"unknown feature mode {mode!r}")
    s7 = _intas_node_features(record, "sender")
    r7 = _intas_node_features(record, "receiver")
    if mode == "gnn7":
        return s7, r7
    edge = float(record["sender"]["distance_to_road_edge"])
    if not math.isfinite(edge):
        raise ValueError("non-finite sender.distance_to_road_edge")
    nan = np.float32("nan")
    if mode == "gnn7_edge":
        return np.append(s7, np.float32(edge)), np.append(r7, nan)
    dx, dy = float(s7[0] - r7[0]), float(s7[1] - r7[1])
    dist = math.hypot(dx, dy)
    dspd, dacl = float(s7[3] - r7[3]), float(s7[4] - r7[4])
    sin_s, cos_s, sin_r, cos_r = (float(v) for v in (s7[5], s7[6], r7[5], r7[6]))
    hdc = cos_s * cos_r + sin_s * sin_r            # cos(heading_s - heading_r)
    hds = sin_s * cos_r - cos_s * sin_r            # sin(heading_s - heading_r)
    sender = np.asarray([s7[3], s7[4], sin_s, cos_s, edge, dx, dy, dist, dspd, dacl, hdc, hds, 1.0], dtype=np.float32)
    receiver = np.asarray([r7[3], r7[4], sin_r, cos_r, nan, -dx, -dy, dist, -dspd, -dacl, hdc, -hds, 0.0],
                          dtype=np.float32)
    return sender, receiver


def audit_intas_splits(train_records, val_records, test_records):
    """Report sender/receiver identity overlap across official splits; never move records."""
    splits = {
        "Train": train_records,
        "Validation": val_records,
        "Test": test_records,
    }
    ids = {}
    for split_name, records in splits.items():
        senders, receivers = set(), set()
        missing_receiver = 0
        for record in records:
            if record.get("sender_id") is not None:
                senders.add(str(record["sender_id"]))
            rid = record.get("receiver_id")
            if rid is None or str(rid).strip() == "":
                missing_receiver += 1
            else:
                receivers.add(str(rid))
        ids[split_name] = (senders, receivers)
        print(f"[identity audit {split_name}] records={len(records):,}; "
              f"unique_senders={len(senders):,}; unique_receivers={len(receivers):,}; "
              f"missing_receiver_id={missing_receiver:,}")

    names = ("Train", "Validation", "Test")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sender_overlap = ids[a][0] & ids[b][0]
            receiver_overlap = ids[a][1] & ids[b][1]
            print(f"[identity overlap {a} vs {b}] senders={len(sender_overlap):,}; "
                  f"receivers={len(receiver_overlap):,}"
                  + (f"; receiver_examples={sorted(receiver_overlap)[:5]}" if receiver_overlap else ""))


def _intas_raw_samples(records, split_name, feature_mode="gnn7"):
    """Convert raw records to per-message arrays, grouped by sender ID."""
    grouped = {}
    bad = 0
    for idx, record in enumerate(records):
        try:
            sender_id = str(record["sender_id"])
            receiver_id = record.get("receiver_id")
            if receiver_id is None or str(receiver_id).strip() == "":
                raise ValueError("missing receiver_id (expected from the official receiver filename)")
            receiver_id = str(receiver_id)
            label = int(record["attacker"])
            if label not in (0, 1):
                raise ValueError(f"Invalid attacker label {label}")
            sender_x, receiver_x = _intas_pair_features(record, feature_mode)
            rcv = int(record["rcvTime"])
            send = int(record["sendTime"])
            # The timestamp magnitude and 1e9-scale beacon intervals indicate nanoseconds.
            # T stores a per-message send-to-receive delay proxy, not a persistent pairwise
            # interaction age; receiver identity is retained separately as metadata.
            delay_seconds = max(0.0, (rcv - send) / 1_000_000_000.0)
            grouped.setdefault(sender_id, []).append(
                (rcv, sender_x, receiver_x, delay_seconds, label, receiver_id)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            bad += 1
            if bad <= 5:
                print(f"[WARN] {split_name} record {idx} skipped: {exc}")
    if bad:
        print(f"[WARN] {split_name}: skipped {bad} invalid records")
    if not grouped:
        raise RuntimeError(f"No valid InTAS records found for {split_name}")
    return grouped


def build_intas_clients(train_records, val_records, test_records, max_samples_per_client=None, seed=42,
                        feature_mode="gnn7"):
    """Build official InTAS splits as Clients; fit feature normalization on Train only.

    Each message becomes a graph with two role nodes: node 0=sender/ego, node 1=receiver.
    Receiver identity is preserved in Client.receiver_ids for auditing/traceability, but is not a feature.
    A is bidirectional for this single message. This is NOT yet a persistent multi-vehicle graph.
    T is the message-delay proxy described above.
    """
    raw = {
        "Train": _intas_raw_samples(train_records, "Train", feature_mode),
        "Validation": _intas_raw_samples(val_records, "Validation", feature_mode),
        "Test": _intas_raw_samples(test_records, "Test", feature_mode),
    }

    # Normalization statistics use only the official training records (both role nodes).
    train_feature_rows = []
    for samples in raw["Train"].values():
        for _, sender_x, receiver_x, _, _, _ in samples:
            train_feature_rows.extend((sender_x, receiver_x))
    fit_x = np.stack(train_feature_rows).astype(np.float64)
    if feature_mode == "gnn7":
        mean = fit_x.mean(axis=0).astype(np.float32)
        std = fit_x.std(axis=0).astype(np.float32)
    else:   # NaN placeholders (receiver road-edge) are ignored in the Train statistics
        mean = np.nanmean(fit_x, axis=0).astype(np.float32)
        std = np.nanstd(fit_x, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    adjacency = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    output = {}
    for split_name, senders in raw.items():
        clients = []
        for sender_id in sorted(senders):
            samples = sorted(senders[sender_id], key=lambda row: row[0])
            if max_samples_per_client is not None and len(samples) > max_samples_per_client:
                # Evenly sample the sender's timeline for deterministic smoke tests/caps.
                chosen = np.linspace(0, len(samples) - 1, max_samples_per_client, dtype=int)
                samples = [samples[i] for i in chosen]

            xs, adj, times, ys, receiver_ids = [], [], [], [], []
            for _, sender_x, receiver_x, delay_seconds, label, receiver_id in samples:
                nodes = np.stack((sender_x, receiver_x), axis=0)
                nodes = ((nodes - mean) / std).astype(np.float32)
                if feature_mode != "gnn7":
                    nodes = np.clip(np.nan_to_num(nodes, nan=0.0), -INTAS_CLIP, INTAS_CLIP).astype(np.float32)
                xs.append(nodes)
                adj.append(adjacency.numpy())
                tmat = np.zeros((2, 2), dtype=np.float32)
                tmat[0, 1] = tmat[1, 0] = float(delay_seconds)
                times.append(tmat)
                ys.append(float(label))
                receiver_ids.append(receiver_id)

            labels = set(ys)
            if len(labels) > 1:
                print(f"[INFO] {split_name} sender {sender_id} has mixed message labels; retaining per-message labels")
            clients.append(Client(
                X=torch.tensor(np.stack(xs), dtype=torch.float32),
                A=torch.tensor(np.stack(adj), dtype=torch.float32),
                T=torch.tensor(np.stack(times), dtype=torch.float32),
                y=torch.tensor(ys, dtype=torch.float32),
                deg=1,
                client_id=str(sender_id),
                receiver_ids=receiver_ids,
            ))
        output[split_name] = clients

    print(f"[InTAS adapter] feature_mode={feature_mode}; features={fit_x.shape[1]}; nodes/message=2; client grouping=sender_id; "
          "receiver_id retained as metadata (not a feature); bootstrap=OFF")
    for name in ("Train", "Validation", "Test"):
        clients = output[name]
        n_samples = sum(len(c) for c in clients)
        n_pos = sum(float(c.y.sum()) for c in clients)
        print(f"[InTAS {name}] clients={len(clients):,}, messages={n_samples:,}, "
              f"benign={n_samples-int(n_pos):,}, malicious={int(n_pos):,}")
    return output["Train"], output["Validation"], output["Test"]


def limit_clients_stratified(clients, limit, seed=42):
    """Deterministically subsample clients within a split for smoke testing only."""
    if limit is None or len(clients) <= limit:
        return clients
    rng = np.random.default_rng(seed)
    buckets = {0: [], 1: []}
    for c in clients:
        label = int(float(c.y.mean()) >= 0.5)
        buckets[label].append(c)
    for label in buckets:
        rng.shuffle(buckets[label])
    picked = []
    # Alternate classes to keep both represented when possible.
    while len(picked) < limit and (buckets[0] or buckets[1]):
        for label in (0, 1):
            if buckets[label] and len(picked) < limit:
                picked.append(buckets[label].pop())
    return picked


# ----------------------------------------------------------------------------------------------
# 8. Self-tests and CLI
# ----------------------------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    # clipping radius: monotone in degree, <= Cmax, cold-start floor, uniform baseline
    cfg = DPConfig(alpha=1.0, cmax=3.0, tau_min=3)
    cs = [clip_radius(d, cfg, rng) for d in range(1, 40)]
    assert all(c <= cfg.cmax + 1e-9 for c in cs) and all(b >= a for a, b in zip(cs, cs[1:]))
    assert clip_radius(1, cfg, rng) >= math.log1p(cfg.tau_min) - 1e-9
    assert clip_radius(5, DPConfig(mode="uniform"), rng) == 3.0
    # neighbouring datasets (same topology) -> same C_i; privatized radius is continuous and in [Cmin, Cmax]
    assert clip_radius(7, cfg, rng) == clip_radius(7, cfg, rng)
    pc = DPConfig(mode="privatized", sigma_deg=2.0)
    assert all(math.log1p(3) - 1e-9 <= clip_radius(d, pc, rng) <= 3.0 + 1e-9 for d in range(0, 30))
    # clipping bound holds
    g = torch.Generator().manual_seed(0)
    v = torch.randn(1000) * 5
    assert float(clip_and_noise(v, 2.0, 0.0, g).norm()) <= 2.0 + 1e-5
    # hierarchical == flat weighted average
    ups = [torch.randn(10) for _ in range(7)]
    sizes = [3, 5, 2, 8, 1, 9, 4]
    rsus = [0, 1, 0, 2, 1, 2, 0]
    flat = sum(s / sum(sizes) * u for s, u in zip(sizes, ups))
    assert torch.allclose(hierarchical_aggregate(ups, sizes, rsus, 3), flat, atol=1e-6)
    # Accountant: epsilon decreases with sigma, increases with rounds; calibration meets target.
    assert _rdp_step(0.05, 4.0, 8) < _rdp_step(0.05, 2.0, 8)
    a, b = RDPAccountant(0.05), RDPAccountant(0.05)
    a.step([2.0], 50); b.step([4.0], 50)
    assert a.epsilon() > b.epsilon()
    s = calibrate_sigma(1.0, 0.05, 78)
    acc = RDPAccountant(0.05)
    acc.step([s], 78)
    assert acc.epsilon() <= 1.001, acc.epsilon()
    # Central DP aggregation clips every contribution and uses one server noise vector.
    ref = torch.zeros(4)
    upd, noise_norm = central_dp_aggregate(
        [torch.tensor([10., 0., 0., 0.]), torch.tensor([0., 10., 0., 0.])],
        [0, 1], 2, ref, 1.0, 0.0, 2.0, torch.Generator().manual_seed(0)
    )
    assert torch.allclose(upd, torch.tensor([0.5, 0.5, 0.0, 0.0]))
    assert noise_norm == 0.0
    # lambda is part of the flattened parameter vector (clipped / noised / aggregated with the rest)
    m = TemporalGNN(6, 8, 2)
    n_lam = sum(p.numel() for p in m.parameters())
    assert any(p is m.lam_hat for p in m.parameters()) and n_lam > 1
    X, A, T, y = pool(make_synthetic(10, 3))
    assert m(X, A, T).shape == y.shape
    print("selftest OK  (sigma for eps=1, q=0.05, R=78: %.3f)" % s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", default="topology", choices=["topology", "uniform", "privatized", "none"])
    ap.add_argument("--epsilon", type=float, default=float("inf"))
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--vehicles", type=int, default=600)
    ap.add_argument("--clients_per_round", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=16, help="paper: 64; small models suit client-level DP")
    ap.add_argument("--layers", type=int, default=2, help="paper: 3")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cmax", type=float, default=3.0)
    ap.add_argument("--tau_min", type=int, default=3)
    ap.add_argument("--no_bootstrap", action="store_true")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--local_epochs", type=int, default=5)
    ap.add_argument("--pos_weight", type=float, default=2.0)
    ap.add_argument("--server_lr", type=float, default=None, help="default: 1.0 without DP, 0.25 with DP")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=None, help="folder with VeReMi-Extension receiver JSON files (veh_*.json)")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=500, help="max ego graphs per VeReMi receiver file")
    ap.add_argument("--stride", type=int, default=0, help="use every n-th beacon as a sample (0 = auto)")
    ap.add_argument("--min_samples", type=int, default=8, help="drop receiver files with fewer ego graphs")
    ap.add_argument("--split", default="sender", choices=["sender", "receiver"],
                    help="VeReMi split: sender=vehicle-disjoint, receiver=file split")
    ap.add_argument("--intas", action="store_true", help="use official InTAS Train/Validation/Test from spliteDataset.py")
    ap.add_argument("--smoke_test", action="store_true", help="tiny InTAS integration test (2 rounds; not for reporting metrics)")
    ap.add_argument("--max_clients_per_split", type=int, default=None, help="optional per-split client cap; intended for smoke tests")
    ap.add_argument("--max_samples_per_client", type=int, default=None, help="optional message cap per sender client")
    ap.add_argument("--intas_features", default="gnn7", choices=list(INTAS_FEATURE_MODES),
                    help="gnn7 = original; edge_rel_kin = location-free + road-edge distance; gnn7_edge = original + road-edge")
    ap.add_argument("--centralized_epochs", type=int, default=0,
                    help="ladder step 1: train the same GNN on pooled Train data (no federation/DP) for N epochs")
    ap.add_argument("--central_bs", type=int, default=64, help="batch size for --centralized_epochs")
    ap.add_argument("--report_test", action="store_true",
                    help="InTAS only: evaluate the official Test split (default: Test is NOT evaluated)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.intas and a.data_dir:
        ap.error("Use --intas or --data_dir, not both")
    if a.smoke_test and not a.intas:
        ap.error("--smoke_test is intended for InTAS; add --intas")

    if a.smoke_test:
        # Deliberately small and non-private: only validate shapes, training, and evaluation.
        a.mode = "none"
        a.epsilon = float("inf")
        a.rounds = 2
        a.clients_per_round = 8
        a.hidden = 8
        a.layers = 1
        a.local_epochs = 1
        a.lr = 0.05
        a.no_bootstrap = True
        a.max_clients_per_split = a.max_clients_per_split or 24
        a.max_samples_per_client = a.max_samples_per_client or 8
        print("[SMOKE TEST] small non-private run; metrics are diagnostic only, not research results")

    mob = (0, 1, 2, 3)
    is_intas = bool(a.intas)
    if is_intas:
        from spliteDataset import load_dataset
        train_records, val_records, test_records = load_dataset()
        audit_intas_splits(train_records, val_records, test_records)
        tr, va, te = build_intas_clients(
            train_records, val_records, test_records,
            max_samples_per_client=a.max_samples_per_client, seed=a.seed,
            feature_mode=a.intas_features
        )
        tr = limit_clients_stratified(tr, a.max_clients_per_split, a.seed)
        va = limit_clients_stratified(va, a.max_clients_per_split, a.seed + 1)
        te = limit_clients_stratified(te, a.max_clients_per_split, a.seed + 2)
        print(f"[official InTAS split; no records moved] clients used train/val/test: {len(tr)}/{len(va)}/{len(te)}")
        mob = ()
    elif a.data_dir:
        from veremi_loader import load_veremi_dir, split_by_sender, MOB_IDX
        clients = load_veremi_dir(a.data_dir, max_files=a.max_files, stride=a.stride,
                                  max_samples=a.max_samples, min_samples=a.min_samples, seed=a.seed)
        tr, va, te = split_by_sender(clients, a.seed) if a.split == "sender" else split_vehicles(clients, a.seed)
        mob = MOB_IDX
        print(f"[split] train/val/test receivers: {len(tr)}/{len(va)}/{len(te)}")
    else:
        clients = make_synthetic(a.vehicles, seed=a.seed)
        tr, va, te = split_vehicles(clients, a.seed)

    if not tr or not va or not te:
        raise SystemExit("Train, Validation, and Test must all contain at least one client.")
    if a.intas and a.smoke_test:
        for split_name, clients in (("Train", tr), ("Validation", va), ("Test", te)):
            n = sum(len(c) for c in clients)
            labels = torch.cat([c.y for c in clients])
            print(f"[smoke {split_name}] X=({n}, 2, {clients[0].X.shape[-1]}), A=({n}, 2, 2), T=({n}, 2, 2), y=({n},)")
            print(f"[smoke {split_name}] class counts: benign={int((labels == 0).sum())}, malicious={int((labels == 1).sum())}")

    cfg = FedConfig(
        mob_idx=mob, lr=a.lr, local_epochs=a.local_epochs, pos_weight=a.pos_weight,
        server_lr=(a.server_lr if a.server_lr is not None else
                   (1.0 if (a.mode == 'none' or math.isinf(a.epsilon)) else 0.25)),
        rounds=a.rounds, clients_per_round=min(a.clients_per_round, len(tr)),
        epsilon=a.epsilon, delta=a.delta, hidden=a.hidden, layers=a.layers,
        seed=a.seed, use_bootstrap=(not a.no_bootstrap and not is_intas),
        dp=DPConfig(mode=a.mode, alpha=a.alpha, cmax=a.cmax, tau_min=a.tau_min)
    )
    if is_intas:
        print("[InTAS] Bootstrap is disabled: receiver IDs are metadata only; graph is still per-message.")
    if a.centralized_epochs > 0:
        if not (a.mode == "none" or math.isinf(a.epsilon)):
            ap.error("--centralized_epochs is a non-private ladder step; use --epsilon inf")
        print(f"[ladder step 1] centralized training: epochs={a.centralized_epochs}, bs={a.central_bs}, "
              f"hidden={a.hidden}, layers={a.layers}, lr={a.lr}")
        model, boot, eps = train_centralized(cfg, tr, va, a.centralized_epochs, bs=a.central_bs)
    else:
        model, boot, eps = train(cfg, tr, va)
    val_logits, val_y = predict_logits(model, va, boot, cfg)

    # Diagnostic only: report continuous score distributions. This does not train the model,
    # alter the DP mechanism, or select a threshold using Test labels.
    report_test = (not is_intas) or a.report_test
    report_score_distribution("Validation", val_logits, val_y)
    if report_test:
        test_logits, test_y = predict_logits(model, te, boot, cfg)
        report_score_distribution("Test", test_logits, test_y)

    # Select the decision threshold on Validation only. Test is not used for model selection.
    threshold, val_metrics = select_threshold(val_logits, val_y)
    print("VALIDATION (threshold selected here):",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()})
    print(f"[threshold] selected on Validation only: {threshold:.6f}")

    if not report_test:
        print("[TEST NOT EVALUATED] InTAS Test is evaluated only with --report_test, after configuration is frozen.")
        return
    # Evaluate the official Test split only after the threshold has been fixed. Reuse the
    # already-computed logits so the diagnostic does not cause a duplicate forward pass.
    m = metrics(test_logits, test_y, threshold=threshold)
    label = "TEST (official InTAS split; message-level)" if is_intas else "TEST (vehicle-disjoint)"
    print(label + ":", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
          f"| eps spent = {eps:.2f}")




if __name__ == "__main__":
    main()
