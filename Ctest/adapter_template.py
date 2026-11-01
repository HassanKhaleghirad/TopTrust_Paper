"""Template: connect TopTrust's PyTorch-Geometric model and data to the harness.
Run:  python run_experiments.py --adapter adapter_template:build
Flatten parameters with torch.nn.utils.parameters_to_vector / vector_to_parameters."""
import numpy as np
from fl import Client, Model

class TopTrustModel(Model):
    def __init__(self, net, device="cpu"): self.net, self.device = net, device
    def init_params(self, rng):
        raise NotImplementedError("return flat numpy vector of ALL parameters incl. lambda_hat")
    def local_update(self, params, data, epochs, lr, rng):
        raise NotImplementedError("load params, train E epochs on this vehicle's temporal ego graphs, return new - old")
    def evaluate(self, params, test):
        raise NotImplementedError("return dict(f1=..., precision=..., recall=..., mcc=..., auc=...) on the vehicle-disjoint test split")
    def per_sample_loss(self, params, data):
        raise NotImplementedError("per-graph cross-entropy loss, used by the membership-inference attack")

def build():
    """Return (model, clients, test, cfg_overrides). Each Client needs: cid, deg (public degree), rsu (RSU index),
    data (your container), n (number of local samples). Use the vehicle-disjoint 70/15/15 split of the paper:
    clients = training vehicles only; `test` = test vehicles."""
    raise NotImplementedError
    # cfg_overrides example: dict(q=50/N, rounds=78, epochs=5, lr=..., alpha=1.0, c_max=..., tau_min=..., seeds=5)
