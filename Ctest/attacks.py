"""Byzantine / colluding attackers acting on raw local updates (before clipping), plus degree inflation.
Attackers are omniscient about the other selected vehicles' raw updates (a strong, conservative assumption)."""
import numpy as np

KINDS = ("signflip", "scale", "gauss", "alie", "aligned")

class Attack:
    def __init__(self, kind, scale=10.0, alie_z=1.0):
        assert kind in KINDS
        self.kind, self.scale, self.alie_z = kind, scale, alie_z
    def apply(self, sel, raw, rng):
        byz = [c for c in sel if c.byz]
        if not byz: return raw
        out = dict(raw)
        allu = np.stack([raw[c.cid] for c in sel])
        mu, sd = allu.mean(0), allu.std(0)
        direction = -mu / (np.linalg.norm(mu) + 1e-12)
        for c in byz:
            u = raw[c.cid]
            if self.kind == "signflip": out[c.cid] = -self.scale * u               # reversed (and scaled) update
            elif self.kind == "scale": out[c.cid] = self.scale * u                 # inflated update
            elif self.kind == "gauss": out[c.cid] = rng.normal(0.0, self.scale, size=u.shape)
            elif self.kind == "alie": out[c.cid] = mu - self.alie_z * sd           # "a little is enough" (collusion)
            elif self.kind == "aligned": out[c.cid] = 1e3 * direction              # all colluders push the same direction
        return out

def make_byzantine(clients, frac, rng, inflate_degree=False):
    """Mark a fraction of vehicles malicious. With inflate_degree they announce the largest observed degree,
    which raises their clipping bound to C_max (worst case for degree-dependent clipping)."""
    k = int(round(frac * len(clients)))
    idx = rng.choice(len(clients), size=k, replace=False)
    dmax = max(c.deg for c in clients)
    for i in idx:
        clients[i].byz = True
        clients[i].fake_deg = dmax if inflate_degree else None
    return clients

def reset_byzantine(clients):
    for c in clients: c.byz, c.fake_deg = False, None
