"""Negative (teeth) test for the S2.26 analytic closed-form check: mutate the
composer's recurrence to the fla k(x)v convention (the exact S2.25 bug class)
and confirm analytic_closed_form_check's assertion FIRES."""
import sys

import torch

sys.path.insert(0, "/Users/samuellarson/Experiments/learned-representations/matrix-thinking/capability_separation")
import stage2_composer as sc


def mutated_states_from_embedding(self, tok_embed, reset_every=None):
    """The fla-convention (transposed) update: S = (I - b kk^T) S + b k v^T."""
    if self.widen is not None:
        tok_embed = tok_embed + self.widen(tok_embed)
    B, D, h = tok_embed.shape
    n_h = self.n_h
    k = self.k_proj(tok_embed).view(B, D, n_h, h)
    v = self.v_proj(tok_embed).view(B, D, n_h, h)
    k = torch.nn.functional.normalize(k, dim=-1, eps=1e-8)
    beta = self.beta_max * torch.sigmoid(self.beta_proj(tok_embed).view(B, D, n_h))
    eye = torch.eye(h, device=tok_embed.device, dtype=tok_embed.dtype).unsqueeze(0)
    S = tok_embed.new_zeros(B, h, h)
    states = [S]
    for t in range(D):
        for j in range(n_h):
            kk = k[:, t, j, :]
            vv = v[:, t, j, :]
            bb = beta[:, t, j].view(B, 1, 1)
            kkT = kk.unsqueeze(-1) @ kk.unsqueeze(-2)
            S = (eye - bb * kkT) @ S + bb * (kk.unsqueeze(-1) @ vv.unsqueeze(-2))  # MUTANT
        states.append(S)
    return states


orig = sc.GroupWordDeltaComposer.states_from_embedding
sc.GroupWordDeltaComposer.states_from_embedding = mutated_states_from_embedding
try:
    sc.analytic_closed_form_check(device="cpu")
except AssertionError as e:
    print(f"\nMUTANT KILLED (assertion fired as required): {e}")
    sc.GroupWordDeltaComposer.states_from_embedding = orig
    sc.analytic_closed_form_check(device="cpu")
    print("\nRestored composer passes again. TEETH CONFIRMED.")
    sys.exit(0)
print("\nFATAL: the mutant SURVIVED the analytic check -- the check has NO TEETH")
sys.exit(1)
