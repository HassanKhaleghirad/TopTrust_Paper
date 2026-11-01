"""RDP accountant for the Poisson-sampled Gaussian mechanism (uses Opacus' analysis code).
Install with `pip install opacus`; alternatively set RDP_PY to a local copy of opacus/accountants/analysis/rdp.py."""
import importlib.util, os

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def _rdp():
    try:
        from opacus.accountants.analysis import rdp
        return rdp
    except Exception:
        path = os.environ.get("RDP_PY")
        if not path:
            raise ImportError("Install opacus (pip install opacus) or set RDP_PY to rdp.py")
        spec = importlib.util.spec_from_file_location("rdp_local", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

def eps_sampled_gaussian(q, z, steps, delta):
    m = _rdp()
    rdp = m.compute_rdp(q=q, noise_multiplier=z, steps=steps, orders=ORDERS)
    return m.get_privacy_spent(orders=ORDERS, rdp=rdp, delta=delta)[0]

def calibrate_z(target_eps, q, steps, delta, lo=0.3, hi=60.0, tol=1e-3):
    """Smallest noise multiplier z whose vehicle-level epsilon is <= target_eps (bisection)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if eps_sampled_gaussian(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi
