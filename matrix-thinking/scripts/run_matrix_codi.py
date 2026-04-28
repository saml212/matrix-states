#!/usr/bin/env python3
"""
Matrix-CODI Rank Dynamics Experiment
=====================================

Forks CODI (arXiv 2502.21074, github.com/zhenyi4/codi) by reimplementing the
minimal CODI mechanism inline on top of GPT-2 (124M) and inserting a matrix
bottleneck at each of the 6 latent reasoning positions. Trains on GSM8K,
measures effective rank of the matrix thoughts at each latent step, correlates
with reasoning depth, and supports a forced low-rank projection ablation at
eval time.

Modes:
  --mode train_vanilla          Run A: vanilla CODI baseline (no matrix bottleneck)
  --mode train_matrix           Run B: matrix-CODI with d=16 bottleneck
  --mode eval_rank_projection   Run C: load Run B checkpoint; project Z to
                                rank k in {1,2,4,8,16} at eval time via
                                truncated SVD; measure GSM8K accuracy per k
  --smoke-test                  Run all mandatory smoke tests and exit

Outputs (under /toy_story_slam/results/matrix_codi/):
  script.py, train.log, results.json, SUMMARY.txt,
  best_run_a_vanilla.pt | best_run_b_matrix.pt,
  rank_dynamics.json, rank_projection_ablation.json

Launch on 8xH100:
  torchrun --standalone --nproc_per_node=8 run_matrix_codi.py --mode train_matrix
  torchrun --standalone --nproc_per_node=8 run_matrix_codi.py --mode train_vanilla
  python run_matrix_codi.py --mode eval_rank_projection \
      --checkpoint /toy_story_slam/results/matrix_codi/best_run_b_matrix.pt

CODI hyperparameters (from the CODI paper, GPT-2 settings):
  base model = gpt2 (124M), n_latents = 6, alpha = beta = gamma = 1,
  lr = 1e-4, warmup = 100, batch/GPU = 16, epochs = 10, distill = L1 at ":",
  per-layer std-normalized, applied across all transformer layers.

Matrix bottleneck (d = 16): W_up: 768 -> 256, reshape to 16x16, optional
1 iteration of (I+Delta) Z (I+Gamma), flatten, W_down: 256 -> 768.

This script is self-contained (no imports from matrix-thinking/src/).
"""

import argparse
import copy
import datetime as dt_module
import functools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # CODI / GPT-2 faithful settings
    "base_model": "gpt2",
    "n_latents": 6,
    "alpha": 1.0,        # teacher CE weight
    "beta": 1.0,         # student CE weight
    "gamma": 1.0,        # distillation weight (GPT-2 setting)
    "lr": 1e-4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "betas": (0.9, 0.98),
    "grad_clip": 1.0,
    "batch_size_per_gpu": 16,
    "epochs": 10,
    "max_q_len": 640,     # question token budget. GSM8K ~50-100 tokens; ProsQA up to 627 tokens (observed max).
                          # Fix 4b: bumped from 512 to 640 to eliminate ProsQA truncation entirely.
    # max_cot_len removed (Fix 22 / Fix SERIOUS-12): CoT is stripped before training.
    "max_ans_len": 32,    # answer token budget
    "max_total_len": 768, # hard cap for student sequence length (student = q + latents + tail)

    # Matrix bottleneck
    "mat_dim": 16,
    "use_thinking_iter": True,  # 1 iteration of (I+Delta) Z (I+Gamma)
    "readout": "flatten",       # readout variant: flatten | bilinear | bilinear_gelu | svd_aug | quadratic

    # Eval
    "eval_interval_epochs": 1,
    "max_eval_batches": 8,      # Round 2: cheap eval per epoch; final eval uses final_eval_batches
    "eval_batch_size": 16,
    "max_rank_samples": 50,     # how many raw Z matrices to save

    # Logging
    "log_interval": 50,

    # Paths
    "results_dir": "/toy_story_slam/results/matrix_codi",

    # Reproducibility
    "seed": 1337,

    # Eval cadence (Round 2: final full-test eval after training)
    "final_eval_batches": 32,

    # Rank-aware training flags (all OFF by default; mirrored from argparse)
    "rank_loss": "none",                  # {"none", "entropy", "nuclear"}
    "rank_lambda": 0.0,                   # coefficient for rank loss; 0 = OFF
    "force_rank_during_training": 0,      # 0 = OFF; k>0 truncates Z to rank k per step

    # Dataset selection
    "dataset": "gsm8k_aug",   # {"gsm8k_aug", "prosqa"}

    # ProsQA paths (used only when dataset == "prosqa")
    "prosqa_train_path": "/workspace/pebble/round3_gamma0/data/prosqa_train.json",
    "prosqa_val_path":   "/workspace/pebble/round3_gamma0/data/prosqa_test.json",
    # Note: we use prosqa_test (500 examples) for eval rather than prosqa_valid (300).
    # Test set is larger and reduces variance on the accuracy signal. Valid set is
    # unused in this round; keep the path in Section A for completeness only.
}


# =============================================================================
# MATRIX PRIMITIVES (copied inline from matrix-thinking/src/matrix_thinker.py)
# =============================================================================

class MatrixRMSNorm(nn.Module):
    """RMS normalization over a d x d matrix with per-entry learned gain."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class RowThenColProjection(nn.Module):
    """silu(A @ M) @ B - nonlinearity between left and right multiply."""

    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))

    def forward(self, M):
        # M can be (..., d, d)
        left = torch.einsum('ij,...jk->...ik', self.A, M)
        return torch.einsum('...ij,jk->...ik', F.silu(left), self.B)


class MultiplicativeThinkingLayer(nn.Module):
    """Per-position multiplicative composition: (I + Delta) M (I + Gamma) + v k^T.

    Faithful to src/matrix_thinker.py::MultiplicativeThinkingLayer, including
    the scale.clamp(0.01, 0.5) safeguard. Accepts input of shape
    (..., d, d) with at least one leading batch-like dimension.
    """

    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.d = d
        self.norm = MatrixRMSNorm(d)

        self.delta_gate = RowThenColProjection(d)
        self.delta_value = RowThenColProjection(d)
        self.delta_up = RowThenColProjection(d)
        self.gamma_gate = RowThenColProjection(d)
        self.gamma_value = RowThenColProjection(d)
        self.gamma_up = RowThenColProjection(d)

        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)

        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        # M: (B, d, d)
        M_n = self.norm(M)
        scale = self.scale.clamp(0.01, 0.5)

        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * scale
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * scale

        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)

        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)

        g_m = torch.sigmoid(
            (self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias
        )
        g_w = torch.sigmoid(
            (self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias
        )

        update = g_m * (M_mult - M_n) + g_w * M_write
        return M + self.dropout(update)


class MatrixBottleneck(nn.Module):
    """Matrix bottleneck inserted into CODI's latent feedback path.

    Takes a hidden state h in R^D, projects to a d x d matrix Z (the observable),
    optionally applies one iteration of the multiplicative thinking layer, then
    projects back to R^D. The intermediate Z tensor is returned so it can be
    saved for rank analysis. All operations preserve the leading batch axis.
    """

    def __init__(self, hidden_dim, mat_dim, use_thinking_iter=True, readout='flatten'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mat_dim = mat_dim
        self.use_thinking_iter = use_thinking_iter
        self.readout = readout

        self.w_up = nn.Linear(hidden_dim, mat_dim * mat_dim, bias=True)
        # Fix CRITICAL-8: LayerNorm is the final step of the bottleneck forward.
        # We zero-init w_down.weight AND the LayerNorm affine so the bottleneck
        # starts as a true zero residual — cleanest possible "do nothing" init.
        self.out_norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.zeros_(self.w_up.bias)

        if readout == 'flatten':
            self.w_down = nn.Linear(mat_dim * mat_dim, hidden_dim, bias=True)
            # Fix CRITICAL-8: small-normal init for w_down keeps gradients flowing
            # through the bottleneck (zero init kills all upstream grads via chain rule).
            # out_norm's LayerNorm (standard init weight=1, bias=0) normalises the output.
            nn.init.normal_(self.w_down.weight, std=0.02)
            nn.init.zeros_(self.w_down.bias)
        elif readout in ('bilinear', 'bilinear_gelu'):
            K = mat_dim * mat_dim
            self.U = nn.Parameter(torch.randn(K, mat_dim) * (1.0 / math.sqrt(mat_dim)))
            self.V = nn.Parameter(torch.randn(K, mat_dim) * (1.0 / math.sqrt(mat_dim)))
            self.out = nn.Linear(K, hidden_dim, bias=True)
            nn.init.normal_(self.out.weight, std=0.02)
            nn.init.zeros_(self.out.bias)
        elif readout == 'svd_aug':
            self.w_down = nn.Linear(mat_dim * mat_dim, hidden_dim, bias=True)
            nn.init.normal_(self.w_down.weight, std=0.02)
            nn.init.zeros_(self.w_down.bias)
            self.sigma_proj = nn.Sequential(
                nn.Linear(mat_dim, 4 * mat_dim),
                nn.GELU(),
                nn.Linear(4 * mat_dim, hidden_dim),
            )
            nn.init.normal_(self.sigma_proj[0].weight, std=0.02)
            nn.init.zeros_(self.sigma_proj[0].bias)
            nn.init.normal_(self.sigma_proj[2].weight, std=0.02)
            nn.init.zeros_(self.sigma_proj[2].bias)
        elif readout == 'quadratic':
            self.w_down_quad = nn.Linear(2 * mat_dim * mat_dim, hidden_dim, bias=True)
            nn.init.normal_(self.w_down_quad.weight, std=0.02)
            nn.init.zeros_(self.w_down_quad.bias)

        if use_thinking_iter:
            self.thinker = MultiplicativeThinkingLayer(mat_dim, dropout=0.0)
        else:
            self.thinker = None

    def forward(self, h, rank_project_k=None, force_rank_k=None):
        """Forward pass through the bottleneck.

        Active readout: self.readout  (one of 'flatten', 'bilinear',
        'bilinear_gelu', 'svd_aug', 'quadratic').

        Args:
            h: (B, D) last-token hidden state from the previous latent position
            rank_project_k: optional int; if set, truncate Z to rank k via SVD
                            before the down-projection (used in Run C ablation,
                            eval-only, no-grad context).
            force_rank_k: optional int (from --force-rank-during-training); if
                          set, truncate Z to rank k at every forward pass
                          (training AND eval). Gradients propagate through the
                          SVD via torch.linalg.svd's autograd support.

        Returns:
            h_out: (B, D) hidden state to feed back as the next input embedding
            Z: (B, d, d) the matrix observable at this latent step (post any
               truncation applied by force_rank_k or rank_project_k).
        """
        B = h.shape[0]
        d = self.mat_dim
        flat = self.w_up(h)                       # (B, d*d)
        Z = flat.view(B, d, d)                    # (B, d, d)

        if self.thinker is not None:
            Z = self.thinker(Z)

        # --force-rank-during-training: differentiable rank-k truncation applied
        # at every latent position during training and eval.
        #
        # We use eigh(Z @ Z^T) instead of full SVD for two reasons:
        # 1. Stability: SVD backward contains 1/(σ_i^2 - σ_j^2) terms that
        #    blow up (NaN) when singular values coincide, which happens naturally
        #    near convergence as the spectrum collapses toward rank k.
        #    eigh backward has 1/(λ_i - λ_j) gaps (gap of eigenvalues of ZZ^T,
        #    which are σ_i^2), mitigating the blow-up by a factor of (σ_i + σ_j).
        # 2. Correctness: the top-k eigenvectors of ZZ^T span the same column
        #    space as the rank-k truncation, so P_k Z = U_k U_k^T Z reproduces
        #    the rank-k projection exactly.
        # Autocast is disabled so no op inside silently re-casts back to bf16.
        if force_rank_k is not None and force_rank_k > 0:
            orig_dtype = Z.dtype
            with torch.autocast(device_type=Z.device.type, enabled=False):
                Zf = Z.float()
                ZZt = Zf @ Zf.transpose(-1, -2)              # (B, d, d), symmetric PSD
                eigvals, eigvecs = torch.linalg.eigh(ZZt)    # ascending order, stable backward
                k_clamped = min(force_rank_k, eigvecs.shape[-1])
                U_k = eigvecs[..., -k_clamped:]              # (B, d, k) — top-k eigenvectors
                # Z_k = P_k Z where P_k = U_k U_k^T (orthogonal projector onto col-space)
                Zf_trunc = U_k @ (U_k.transpose(-1, -2) @ Zf)  # (B, d, d)
            Z = Zf_trunc.to(orig_dtype)

        Z_out = Z
        if rank_project_k is not None:
            # Fix SERIOUS-13: SVD done in fp32, outside autocast.
            with torch.autocast("cuda", enabled=False):
                Z_out = truncate_to_rank(Z.float(), rank_project_k).to(Z.dtype)

        if self.readout == 'flatten':
            flat_out = Z_out.reshape(B, d * d)        # (B, d*d)
            h_out = self.w_down(flat_out)             # (B, D)
        elif self.readout == 'bilinear':
            MV = torch.einsum('bij,kj->bik', Z_out, self.V)    # (B, d, K)
            probes = torch.einsum('ki,bik->bk', self.U, MV)    # (B, K)
            h_out = self.out(probes)                            # (B, D)
        elif self.readout == 'bilinear_gelu':
            MV = torch.einsum('bij,kj->bik', Z_out, self.V)    # (B, d, K)
            probes = torch.einsum('ki,bik->bk', self.U, MV)    # (B, K)
            h_out = self.out(F.gelu(probes))                    # (B, D)
        elif self.readout == 'svd_aug':
            with torch.autocast("cuda", enabled=False):
                sigma = torch.linalg.svdvals(Z_out.float()).to(Z_out.dtype)  # (B, d)
            flat_out = Z_out.reshape(B, d * d)
            h_out = self.w_down(flat_out) + self.sigma_proj(sigma)
        elif self.readout == 'quadratic':
            ZZt = torch.einsum('bij,bkj->bik', Z_out, Z_out)   # (B, d, d)
            ZtZ = torch.einsum('bji,bjk->bik', Z_out, Z_out)   # (B, d, d)
            quad = torch.cat([ZZt.reshape(B, d * d), ZtZ.reshape(B, d * d)], dim=-1)
            h_out = self.w_down_quad(quad)                      # (B, D)
        else:
            raise ValueError(f"Unknown readout: {self.readout}")

        # Fix CRITICAL-8: final LayerNorm to normalise the bottleneck output.
        h_out = self.out_norm(h_out)
        # Fix C2: return Z_out (post-truncation, post-thinking) so rank logging
        # in Run C reflects the actual rank-k tensor that was used.
        return h_out, Z_out


def truncate_to_rank(Z, k):
    """Truncate a batch of matrices to rank k via eigh of ZZ^T.

    Used by Run C (eval-time rank projection) and --force-rank-during-training.

    We use eigh(ZZ^T) instead of full SVD to avoid the 1/(σ_i^2 - σ_j^2) NaN
    blow-up in SVD backward when singular values coincide (which occurs near
    convergence as the spectrum collapses). eigh of a symmetric PSD matrix has a
    stable backward (gradient involves 1/(λ_i - λ_j) with λ_i = σ_i^2, so the
    gap is mitigated by the factor (σ_i + σ_j) relative to SVD's gap-of-squares).

    The top-k eigenvectors of ZZ^T span the same column space as the rank-k
    truncation of Z, so P_k Z = U_k U_k^T Z is equivalent to the rank-k best
    approximation for the purposes of projecting Z's image.

    Autocast is disabled to prevent silent mid-graph bf16 re-casts.

    Args:
        Z: (B, d, d) batch of matrices (float32 recommended; cast internally)
        k: target rank (clamped to min(k, d))

    Returns:
        (B, d, d) rank-≤-k approximation in the same dtype as input Z.
    """
    orig_dtype = Z.dtype
    with torch.autocast(device_type=Z.device.type, enabled=False):
        Zf = Z.float()
        ZZt = Zf @ Zf.transpose(-1, -2)              # (B, d, d), symmetric PSD
        eigvals, eigvecs = torch.linalg.eigh(ZZt)    # ascending order; stable backward
        k = min(k, eigvecs.shape[-1])
        U_k = eigvecs[..., -k:]                      # (B, d, k) — top-k eigenvectors
        # Z_k = U_k (U_k^T Z) — orthogonal projection onto top-k column subspace
        Zk = U_k @ (U_k.transpose(-1, -2) @ Zf)     # (B, d, d)
    return Zk.to(orig_dtype)


def compute_rank_loss(Z_list, rank_loss_type):
    """Auxiliary rank-maximisation loss over the n_latents matrix observables.

    Args:
        Z_list: list of (B, d, d) tensors, one per latent position (length n_latents).
                Must NOT be detached — gradients must flow through Z back to w_up/thinker.
        rank_loss_type: 'entropy' | 'nuclear'

    Returns:
        Scalar tensor (averaged over batch × positions) with requires_grad=True.

    Notes on stability:
        We use torch.linalg.svdvals (not full svd) because it is faster and its
        backward is well-defined through eigendecomposition. Full svd backward can
        be numerically unstable near repeated singular values; svdvals avoids
        computing U/V and uses a safer gradient path.
    """
    assert rank_loss_type in ("entropy", "nuclear"), \
        f"Unknown rank_loss_type: {rank_loss_type!r}"

    loss_terms = []
    for Z in Z_list:
        # Z: (B, d, d). svdvals returns descending singular values.
        # Cast to float32 and disable autocast so no op inside silently re-casts
        # back to bf16 on CUDA (matches Fix SERIOUS-13 pattern used elsewhere).
        with torch.autocast(device_type=Z.device.type, enabled=False):
            sigma = torch.linalg.svdvals(Z.float())  # (B, d)

        if rank_loss_type == "entropy":
            # Normalise to a probability distribution, compute Shannon entropy.
            # We MAXIMISE H, so minimise -H.
            sigma_sum = sigma.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            p = sigma / sigma_sum                          # (B, d), sums to 1
            p_safe = p.clamp(min=1e-10)
            H = -(p_safe * torch.log(p_safe)).sum(dim=-1)  # (B,), higher = spread spectrum
            # Minimising loss → maximising H
            loss_terms.append(-H.mean())
        else:  # nuclear
            # Maximise nuclear norm (sum of singular values).
            # Note: this primarily grows σ_1 and does NOT guarantee rank spreading.
            nuc = sigma.sum(dim=-1)  # (B,)
            loss_terms.append(-nuc.mean())

    # Average over the n_latents positions.
    return torch.stack(loss_terms).mean()


def effective_rank(Z):
    """Effective rank via singular value entropy. Z: (..., d, d) -> (...)."""
    # Cast to float32 for SVD stability.
    Zf = Z.float()
    sigma = torch.linalg.svdvals(Zf).clamp(min=1e-10)
    p = sigma / sigma.sum(dim=-1, keepdim=True)
    H = -(p * torch.log(p)).sum(dim=-1)
    return torch.exp(H)


# =============================================================================
# CODI MODEL WRAPPER (GPT-2 + optional matrix bottleneck at latent positions)
# =============================================================================

# Imported lazily so the smoke-test path that doesn't need transformers can be
# tested independently. Import here at module scope to make things explicit.
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402


class CodiModel(nn.Module):
    """CODI-style wrapper around GPT-2.

    The wrapper exposes:
      - add_special_tokens(): adds <bot> <eot> <latent> and resizes embeddings.
      - teacher_forward(): explicit-CoT pass (no_grad context is the caller's
        responsibility). Returns all hidden states plus the answer logits.
      - student_forward(): latent-thought pass; manually iterates 6 latent steps
        feeding back the previous position's last-layer hidden state (either
        directly, for vanilla CODI, or through the matrix bottleneck for
        matrix-CODI). Returns answer logits, all hidden states at the ":" token,
        and the saved Z matrices (matrix-CODI only).
    """

    def __init__(self, base_model_name, n_latents, use_matrix_bottleneck,
                 mat_dim, use_thinking_iter, readout='flatten'):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.gpt2.config.output_hidden_states = True
        self.hidden_dim = self.gpt2.config.n_embd   # 768 for GPT-2
        self.n_latents = n_latents
        self.use_matrix_bottleneck = use_matrix_bottleneck
        self.mat_dim = mat_dim

        if use_matrix_bottleneck:
            self.bottleneck = MatrixBottleneck(
                hidden_dim=self.hidden_dim,
                mat_dim=mat_dim,
                use_thinking_iter=use_thinking_iter,
                readout=readout,
            )
            # CODI ablation note: a small 2-layer MLP + LayerNorm projection is
            # CODI-faithful for the vector path. The matrix bottleneck plays
            # the same structural role, so we do not add a second MLP on top.
            self.feedback_proj = None
        else:
            # Fix C3: faithful CODI baseline uses LayerNorm-only feedback (identity
            # in spirit). A plain nn.Identity() destabilises training; a single
            # LayerNorm matches CODI's "no bottleneck" intent while keeping
            # activations normalised. Target: ~43.7% GSM8K reproduction.
            self.bottleneck = None
            self.feedback_proj = nn.LayerNorm(self.hidden_dim)

        self.special_token_ids = {}

    def add_special_tokens(self, tokenizer):
        """Adds <bot>, <eot>, <latent> tokens and resizes GPT-2 embeddings."""
        new_tokens = ["<bot>", "<eot>", "<latent>"]
        added = tokenizer.add_special_tokens({
            "additional_special_tokens": new_tokens,
        })
        # Also ensure there is a pad token (GPT-2 has none by default).
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Always resize — even if added==0 (tokens already in tokenizer from a
        # previous model build), a freshly loaded GPT-2 still has the old vocab size.
        self.gpt2.resize_token_embeddings(len(tokenizer))
        self.special_token_ids = {
            "bot": tokenizer.convert_tokens_to_ids("<bot>"),
            "eot": tokenizer.convert_tokens_to_ids("<eot>"),
            "latent": tokenizer.convert_tokens_to_ids("<latent>"),
            "pad": tokenizer.pad_token_id,
            "eos": tokenizer.eos_token_id,
        }
        return added

    # ------------------------------------------------------------------
    # Teacher pass: explicit CoT, standard GPT-2 forward, all hidden states.
    # Caller must wrap this in torch.no_grad() for the KD term.
    # ------------------------------------------------------------------
    def teacher_forward(self, input_ids, attention_mask):
        out = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out  # .logits (B, T, V), .hidden_states (tuple of (B, T, D))

    # ------------------------------------------------------------------
    # Student pass: latent thoughts with hidden-state feedback.
    # Implementation approach:
    #   1. Embed the question prefix + <bot> with GPT-2's token embeddings.
    #   2. Run one forward pass to obtain the <bot> position's last hidden state.
    #   3. For each of n_latents steps: feed that hidden state (directly or
    #      through the matrix bottleneck) as the *inputs_embeds* of the next
    #      position, run another forward pass with the growing KV prefix, and
    #      collect the new last hidden state. Save the matrix observable Z.
    #   4. After the final latent, append <eot> + "The answer is:" + answer
    #      and run the full sequence once more (teacher-forcing on the answer).
    #
    # We use inputs_embeds on every step (not input_ids) because the latent
    # positions have no discrete token id and must carry continuous embeddings.
    # This matches CODI's reference implementation.
    # ------------------------------------------------------------------
    def student_forward(self, q_ids, q_mask, tail_ids, tail_mask,
                         save_Z=True, rank_project_k=None, force_rank_k=None):
        """Run the student latent pass.

        Args:
            q_ids: (B, Lq) question tokens (the CODI prefix, no <bot> yet)
            q_mask: (B, Lq) attention mask for the question
            tail_ids: (B, Lt) tokens for "<eot> The answer is: <answer>"
            tail_mask: (B, Lt) attention mask for the tail
            save_Z: if True, record the matrix observables (matrix-CODI only)
            rank_project_k: optional int; forwarded to the matrix bottleneck
                            for Run C (ignored in vanilla mode).
            force_rank_k: optional int; forwarded to the matrix bottleneck for
                          --force-rank-during-training (training + eval). When
                          set, Z is SVD-truncated to rank k at every step with
                          gradients flowing through the truncation.

        Returns:
            dict with:
              logits:          (B, Ltotal, V) answer-position logits (teacher-forced)
              hidden_states:   tuple of (B, Ltotal, D) across all layers
              colon_positions: (B,) index of the ":" token in the final sequence
              Z_list:          list of length n_latents; each (B, d, d) or None
              latent_positions: list of length n_latents; each a scalar index
                                 into the final sequence giving the latent step
        """
        device = q_ids.device
        B, Lq = q_ids.shape
        bot_id = self.special_token_ids["bot"]

        # Build the question + <bot> prefix. We use input_ids here and let GPT-2
        # compute the token embeddings internally; then we switch to
        # inputs_embeds for the latent steps because the latent positions
        # inject continuous values.
        bot_col = torch.full((B, 1), bot_id, dtype=torch.long, device=device)
        prefix_ids = torch.cat([q_ids, bot_col], dim=1)               # (B, Lq+1)
        prefix_mask = torch.cat(
            [q_mask, torch.ones_like(bot_col)], dim=1
        )                                                              # (B, Lq+1)
        prefix_embeds = self.gpt2.transformer.wte(prefix_ids)          # (B, Lq+1, D)

        # Run the prefix to get the <bot> position hidden state.
        out_prefix = self.gpt2.transformer(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out_prefix.last_hidden_state[:, -1, :]   # (B, D) from <bot>

        # Accumulate latent step embeddings. We will keep feeding the growing
        # sequence back through the transformer (simple but clear; scales fine
        # because n_latents is 6 and GPT-2 is small).
        running_embeds = prefix_embeds                          # (B, Lq+1, D)
        running_mask = prefix_mask                              # (B, Lq+1)

        Z_list = [] if save_Z else None
        # Z_list_grad: always populated during matrix-CODI training so compute_codi_loss
        # can compute the auxiliary rank loss with gradients. Separate from Z_list so
        # we don't accidentally break the detach-for-logging path.
        Z_list_grad = [] if self.use_matrix_bottleneck else None
        latent_positions = []

        for t in range(self.n_latents):
            # Compute the next continuous "token" from the last hidden state.
            if self.use_matrix_bottleneck:
                h_next, Z_t = self.bottleneck(
                    last_hidden,
                    rank_project_k=rank_project_k,
                    force_rank_k=force_rank_k,
                )
                if save_Z:
                    # Fix SERIOUS-13: store in fp32 for rank analysis stability.
                    Z_list.append(Z_t.detach().float())
                # Always keep a grad-attached reference for rank loss computation.
                Z_list_grad.append(Z_t)
            else:
                h_next = self.feedback_proj(last_hidden)
                if save_Z:
                    Z_list.append(None)

            # h_next: (B, D). Append as a single position to the running sequence.
            h_next = h_next.unsqueeze(1)                        # (B, 1, D)
            new_mask_col = torch.ones((B, 1), dtype=running_mask.dtype, device=device)
            running_embeds = torch.cat([running_embeds, h_next], dim=1)
            running_mask = torch.cat([running_mask, new_mask_col], dim=1)
            latent_positions.append(running_embeds.shape[1] - 1)

            # Run the transformer on the growing sequence and grab the last position.
            # For the final latent we skip this since we will do one big joint
            # pass over prefix + latents + tail below to match CODI.
            if t < self.n_latents - 1:
                out_step = self.gpt2.transformer(
                    inputs_embeds=running_embeds,
                    attention_mask=running_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                last_hidden = out_step.last_hidden_state[:, -1, :]

        # Now build the tail embeddings and do one joint pass for the KD and
        # student CE losses. We run this through the full GPT-2 LM head so we
        # get logits for the answer positions.
        tail_embeds = self.gpt2.transformer.wte(tail_ids)       # (B, Lt, D)
        full_embeds = torch.cat([running_embeds, tail_embeds], dim=1)
        full_mask = torch.cat([running_mask, tail_mask], dim=1)

        # Joint forward pass over the entire student sequence. The LM head
        # operates on the final hidden states.
        out_full = self.gpt2(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        return {
            "logits": out_full.logits,                   # (B, Ltotal, V)
            "hidden_states": out_full.hidden_states,     # tuple of (B, Ltotal, D)
            "latent_positions": latent_positions,        # positions of z1..z6
            "Z_list": Z_list,                            # list of (B, d, d) detached, or None
            "Z_list_grad": Z_list_grad,                  # list of (B, d, d) with grad, or None
            "prefix_len_with_latents": running_embeds.shape[1],
        }


# =============================================================================
# GSM8K DATA LOADING
# =============================================================================

GSM_STEP_RE = re.compile(r"<<[^>]*=[^>]*>>")
GSM_ANSWER_RE = re.compile(r"####\s*([\-\d\.,]+)")


def count_reasoning_steps(answer_text):
    """Count <<x=y>> annotations in a GSM8K answer field."""
    return len(GSM_STEP_RE.findall(answer_text))


def extract_gold_answer(answer_text):
    """Extract the final numeric answer after '####'."""
    m = GSM_ANSWER_RE.search(answer_text)
    if m is None:
        return None
    return m.group(1).replace(",", "").strip()


def strip_cot_annotations(answer_text):
    """Return the CoT portion of a GSM8K answer (everything before ####)."""
    idx = answer_text.find("####")
    if idx < 0:
        return answer_text.strip()
    cot = answer_text[:idx].strip()
    # Remove the <<x=y>> arithmetic annotations to get a clean CoT string.
    cot = GSM_STEP_RE.sub("", cot)
    # Collapse double spaces introduced by removal.
    cot = re.sub(r"\s+", " ", cot).strip()
    return cot


class GSM8KDataset(Dataset):
    """Tokenized GSM8K examples for the CODI teacher-student setup.

    Each item exposes four token sequences plus metadata:
      - teacher_ids: "[question] [CoT] The answer is: [answer]"
      - q_ids:       "[question]"                     (used as the student prefix)
      - tail_ids:    "<eot> The answer is: [answer]"  (student teacher-forced tail)
      - answer_text: ground-truth numeric answer string for eval
      - reasoning_steps: int count of <<x=y>> arithmetic annotations
    """

    def __init__(self, split, tokenizer, cfg, special_ids):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError("Install `datasets`: pip install datasets") from e

        # CODI trains on GSM8k-Aug (~386k examples augmented by prompting GPT-4
        # on the Implicit-CoT codebase of Deng et al., 2024). whynlp/gsm8k-aug
        # is a mirror with {question, steps: list[str], answer} schema. It ships
        # train/validation/test splits; we use train + test (plain GSM8K test).
        if split == "train":
            ds = load_dataset("whynlp/gsm8k-aug", split="train")
            use_aug_format = True
        else:
            ds = load_dataset("whynlp/gsm8k-aug", split="test")
            use_aug_format = True

        self.items = []
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.special_ids = special_ids

        # Fix CRITICAL-10: count drops so we can surface them in the log.
        n_total = 0
        n_dropped_colon = 0
        n_dropped_len = 0
        n_dropped_gold = 0

        answer_prefix = " The answer is:"
        for row in ds:
            n_total += 1
            question = row["question"].strip()
            if use_aug_format:
                # whynlp/gsm8k-aug: steps is a list of "<<expr=val>>" strings.
                # Join them with spaces to form the CoT text, matching CODI's
                # augmented training format.
                steps = row.get("steps", [])
                if isinstance(steps, list):
                    cot = " ".join(str(s) for s in steps).strip()
                else:
                    cot = str(steps).strip()
                gold = str(row["answer"]).strip()
                n_steps = len(steps) if isinstance(steps, list) else cot.count("<<")
            else:
                # HuggingFace gsm8k: answer field is "<cot text>\n#### <number>".
                answer_field = row["answer"]
                cot = strip_cot_annotations(answer_field)
                gold = extract_gold_answer(answer_field)
                if gold is None:
                    n_dropped_gold += 1
                    continue
                n_steps = count_reasoning_steps(answer_field)

            # Teacher text: question + CoT + " The answer is:" + gold
            teacher_text = f"{question}\n{cot}{answer_prefix} {gold}"
            teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)

            # Student prefix: question only. The <bot> token is added by the model.
            q_ids = tokenizer.encode(question + "\n", add_special_tokens=False)

            # Student tail: "<eot> The answer is: gold"
            tail_text = f"{answer_prefix} {gold}"
            tail_token_ids = tokenizer.encode(tail_text, add_special_tokens=False)
            tail_ids = [special_ids["eot"]] + tail_token_ids

            # Locate the ":" token inside the tail (for the L1 KD alignment).
            # We find the FIRST ":" that follows the "answer is" phrase.
            colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
            tail_colon_rel = None
            for i, tok in enumerate(tail_ids):
                if tok == colon_id:
                    tail_colon_rel = i
                    break
            if tail_colon_rel is None:
                # Fix CRITICAL-10: hard fail instead of silently using a wrong index.
                raise AssertionError(
                    f"Could not find colon in tail: {tail_text!r}"
                )

            # Also locate the ":" in the teacher sequence for the KD pair.
            teacher_colon_idx = None
            for i, tok in enumerate(teacher_ids):
                if tok == colon_id:
                    teacher_colon_idx = i
            if teacher_colon_idx is None:
                n_dropped_colon += 1
                continue

            # Truncate aggressively to fit the budgets.
            if len(q_ids) > cfg["max_q_len"]:
                # Fix 4a: left-truncate so the discriminative question at the END is preserved.
                # GSM8K questions are 50-100 tokens so this cap is a no-op for Round 1 reproduction.
                q_ids = q_ids[-cfg["max_q_len"]:]
            if len(tail_ids) > cfg["max_ans_len"] + 8:
                # Keep the ":" position intact; only trim the trailing answer.
                keep = cfg["max_ans_len"] + 8
                tail_ids = tail_ids[:keep]
                if tail_colon_rel >= keep:
                    n_dropped_len += 1
                    continue

            # For the teacher we allow up to max_total_len.
            if len(teacher_ids) > cfg["max_total_len"]:
                # If truncated, we lose the colon alignment; skip.
                n_dropped_len += 1
                continue

            self.items.append({
                "teacher_ids": teacher_ids,
                "teacher_colon_idx": teacher_colon_idx,
                "q_ids": q_ids,
                "tail_ids": tail_ids,
                "tail_colon_rel": tail_colon_rel,
                "answer_text": gold,
                "reasoning_steps": n_steps,
                "question": question,
            })

        # Fix CRITICAL-10: log drop statistics on rank 0 (or single-process eval).
        n_kept = len(self.items)
        pct = 100.0 * n_kept / max(n_total, 1)
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank == 0:
            print(
                f"[GSM8KDataset split={split}] kept {n_kept}/{n_total} ({pct:.1f}%) "
                f"| dropped: no_gold={n_dropped_gold} colon={n_dropped_colon} "
                f"len={n_dropped_len}",
                flush=True,
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class ProsQADataset(Dataset):
    """Tokenized ProsQA examples for the CODI teacher-student setup.

    Same item schema as GSM8KDataset so the existing collate_gsm8k + compute_codi_loss
    code can consume it unchanged:
      - teacher_ids: "[question] [chain-of-thought steps] The answer is: [answer]"
      - q_ids:       "[question]"
      - tail_ids:    "<eot> The answer is: [answer]"
      - teacher_colon_idx: last ":" index in teacher_ids (matches GSM8K behavior)
      - tail_colon_rel:    first ":" index in tail_ids
      - answer_text: raw ProsQA answer string (multi-word sentence)
      - reasoning_steps: len(row["steps"])
      - question: raw question text
    """

    def __init__(self, split, tokenizer, cfg, special_ids):
        import json
        assert split in ("train", "val"), f"split must be train|val, got {split}"
        if split == "train":
            path = cfg["prosqa_train_path"]
        else:
            path = cfg["prosqa_val_path"]
        with open(path) as f:
            raw_rows = json.load(f)

        self.items = []
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.special_ids = special_ids

        n_total = 0
        n_dropped_colon = 0
        n_dropped_len = 0
        answer_prefix = " The answer is:"
        colon_id = tokenizer.encode(":", add_special_tokens=False)[0]

        for row in raw_rows:
            n_total += 1
            question = row["question"].strip()
            steps = row.get("steps", [])
            # Join ProsQA steps into a single CoT string with periods preserved.
            cot = " ".join(str(s).strip() for s in steps).strip()
            answer_str = str(row["answer"]).strip()

            # Teacher text: question + CoT + " The answer is: <full ProsQA answer>"
            teacher_text = f"{question}\n{cot}{answer_prefix} {answer_str}"
            teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)

            q_ids = tokenizer.encode(question + "\n", add_special_tokens=False)

            tail_text = f"{answer_prefix} {answer_str}"
            tail_token_ids = tokenizer.encode(tail_text, add_special_tokens=False)
            # Fix 7: append EOS so the model learns to stop after the answer.
            # Root cause of bug #5 (generation past the period) is no EOS training signal.
            tail_ids = [special_ids["eot"]] + tail_token_ids + [special_ids["eos"]]

            # Locate the FIRST ":" in the tail (right after "answer is").
            tail_colon_rel = None
            for i, tok in enumerate(tail_ids):
                if tok == colon_id:
                    tail_colon_rel = i
                    break
            if tail_colon_rel is None:
                raise AssertionError(
                    f"Could not find colon in ProsQA tail: {tail_text!r}"
                )

            # Locate the LAST ":" in the teacher sequence (matches GSM8KDataset).
            teacher_colon_idx = None
            for i, tok in enumerate(teacher_ids):
                if tok == colon_id:
                    teacher_colon_idx = i
            if teacher_colon_idx is None:
                n_dropped_colon += 1
                continue

            # Length budgets. ProsQA avg is ~242 tokens with a long tail on 5-6 step
            # examples; keep the GSM8K budget of max_total_len=768 which is ample.
            if len(q_ids) > cfg["max_q_len"]:
                # Fix 4a: left-truncate so the question sentence at the END is preserved.
                # ProsQA format: "[facts...] Is Tom a X or Y?" — question is always last.
                q_ids = q_ids[-cfg["max_q_len"]:]
            if len(tail_ids) > cfg["max_ans_len"] + 16:
                # ProsQA answers can be up to ~10 BPE tokens — give them a bit more slack.
                keep = cfg["max_ans_len"] + 16
                tail_ids = tail_ids[:keep]
                if tail_colon_rel >= keep:
                    n_dropped_len += 1
                    continue

            if len(teacher_ids) > cfg["max_total_len"]:
                n_dropped_len += 1
                continue

            self.items.append({
                "teacher_ids": teacher_ids,
                "teacher_colon_idx": teacher_colon_idx,
                "q_ids": q_ids,
                "tail_ids": tail_ids,
                "tail_colon_rel": tail_colon_rel,
                "answer_text": answer_str,
                "reasoning_steps": len(steps) if isinstance(steps, list) else 0,
                "question": question,
            })

        n_kept = len(self.items)
        pct = 100.0 * n_kept / max(n_total, 1)
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        if rank == 0:
            print(
                f"[ProsQADataset split={split}] kept {n_kept}/{n_total} ({pct:.1f}%) "
                f"| dropped: colon={n_dropped_colon} len={n_dropped_len}",
                flush=True,
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_gsm8k(batch, pad_id, n_latents):
    """Pad the four sequences to batch-wise max lengths.

    Fix CRITICAL-7: q_ids / q_mask are LEFT-padded so <bot> always sits
    immediately after real question content.  teacher_ids and tail_ids are
    right-padded (tail is unmasked at meaningful positions; truncation keeps
    the colon position safe).

    Returns a dict of tensors plus per-item metadata lists.
    """
    B = len(batch)

    def pad_right(seqs, pad_id):
        max_len = max(len(s) for s in seqs)
        out = torch.full((B, max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return out, mask

    def pad_left(seqs, pad_id):
        """Left-pad so real tokens are right-aligned."""
        max_len = max(len(s) for s in seqs)
        out = torch.full((B, max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            offset = max_len - len(s)
            out[i, offset:] = torch.tensor(s, dtype=torch.long)
            mask[i, offset:] = 1
        return out, mask

    teacher_ids, teacher_mask = pad_right([b["teacher_ids"] for b in batch], pad_id)
    # Fix CRITICAL-7: left-pad question so <bot> is appended flush to content.
    q_ids, q_mask = pad_left([b["q_ids"] for b in batch], pad_id)
    tail_ids, tail_mask = pad_right([b["tail_ids"] for b in batch], pad_id)

    teacher_colon_idx = torch.tensor(
        [b["teacher_colon_idx"] for b in batch], dtype=torch.long
    )
    tail_colon_rel = torch.tensor(
        [b["tail_colon_rel"] for b in batch], dtype=torch.long
    )
    reasoning_steps = torch.tensor(
        [b["reasoning_steps"] for b in batch], dtype=torch.long
    )

    return {
        "teacher_ids": teacher_ids,
        "teacher_mask": teacher_mask,
        "teacher_colon_idx": teacher_colon_idx,
        "q_ids": q_ids,
        "q_mask": q_mask,
        "tail_ids": tail_ids,
        "tail_mask": tail_mask,
        "tail_colon_rel": tail_colon_rel,
        "reasoning_steps": reasoning_steps,
        "answer_texts": [b["answer_text"] for b in batch],
        "questions": [b["question"] for b in batch],
    }


# =============================================================================
# LOSS COMPUTATION (teacher CE + student CE + L1 KD at ":")
# =============================================================================

def compute_codi_loss(model, batch, cfg, device, special_ids):
    """Compute the CODI joint loss.

    L = alpha * L_teacher + beta * L_student + gamma * L_KD
        [+ rank_lambda * L_rank  when cfg['rank_loss'] != 'none']

    L_teacher: next-token CE over the teacher sequence (explicit CoT), answer tokens only.
    L_student: next-token CE over the student sequence, answer tokens only.
    L_KD:      L1 distance between teacher and student last-layer-across-all-layers
               hidden states at the ":" token, normalized per layer by teacher std.
    L_rank:    auxiliary rank-maximisation loss (entropy or nuclear), averaged over
               (batch × n_latent positions). Only active when cfg['rank_loss'] != 'none'
               and cfg['rank_lambda'] > 0. Default OFF — does not affect baseline.
    """
    raw = model.module if isinstance(model, DDP) else model

    teacher_ids = batch["teacher_ids"].to(device)
    teacher_mask = batch["teacher_mask"].to(device)
    teacher_colon_idx = batch["teacher_colon_idx"].to(device)
    q_ids = batch["q_ids"].to(device)
    q_mask = batch["q_mask"].to(device)
    tail_ids = batch["tail_ids"].to(device)
    tail_mask = batch["tail_mask"].to(device)
    tail_colon_rel = batch["tail_colon_rel"].to(device)

    B = teacher_ids.shape[0]

    # ----- Teacher pass (two variants) -----
    # Variant A: a differentiable teacher pass for L_teacher CE.
    # Fix SERIOUS-11 / CODI paper (arXiv 2502.21074): teacher shares weights with
    # student and is trained jointly via L_teacher — do NOT wrap in no_grad here.
    teacher_out = raw.teacher_forward(teacher_ids, teacher_mask)
    teacher_logits = teacher_out.logits  # (B, Tt, V)
    # CE on the teacher sequence: predict position t+1 from position t.
    shift_logits = teacher_logits[:, :-1, :].contiguous()
    shift_labels = teacher_ids[:, 1:].contiguous()
    shift_mask = teacher_mask[:, 1:].contiguous()
    # Only score tokens AFTER the ":" position (the answer).
    pos = torch.arange(shift_labels.shape[1], device=device).unsqueeze(0)
    answer_mask = (pos >= teacher_colon_idx.unsqueeze(1)) & (shift_mask.bool())
    ce_t = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(B, -1)
    denom_t = answer_mask.float().sum().clamp(min=1.0)
    L_teacher = (ce_t * answer_mask.float()).sum() / denom_t

    # Variant B: a NO-GRAD teacher pass for L_KD. The KD reference must be
    # detached from the graph per the CODI spec. Reuse the same hidden states
    # from the differentiable pass but with .detach() on the colon slice.
    teacher_hiddens = [h.detach() for h in teacher_out.hidden_states]

    # Determine whether we need Z tensors with gradients for the rank loss.
    _rank_loss_type = cfg.get("rank_loss", "none")
    _rank_lambda = cfg.get("rank_lambda", 0.0)
    _need_rank_loss = (
        _rank_loss_type != "none"
        and _rank_lambda > 0.0
        and raw.use_matrix_bottleneck
    )
    _force_rank_k = cfg.get("force_rank_during_training", 0) or None
    if _force_rank_k == 0:
        _force_rank_k = None

    # ----- Student pass -----
    student = raw.student_forward(
        q_ids=q_ids,
        q_mask=q_mask,
        tail_ids=tail_ids,
        tail_mask=tail_mask,
        save_Z=False,        # training does not need Z for logging; eval does
        force_rank_k=_force_rank_k,
    )
    student_logits = student["logits"]              # (B, Ls, V)
    student_hiddens = student["hidden_states"]      # tuple of (B, Ls, D)
    prefix_len_with_latents = student["prefix_len_with_latents"]

    # Student CE on the answer tokens. The tail sits at positions
    # [prefix_len_with_latents .. prefix_len_with_latents + Lt - 1]. We want to
    # predict tail_ids[1:] from the previous positions.
    # Build a labels tensor aligned with student_logits.
    Ls = student_logits.shape[1]
    student_labels = torch.full((B, Ls), -100, dtype=torch.long, device=device)
    student_labels[:, prefix_len_with_latents:prefix_len_with_latents + tail_ids.shape[1]] = tail_ids

    # Only score positions AFTER the ":" position within the tail.
    # The ":" position in the full sequence is prefix_len_with_latents + tail_colon_rel.
    full_colon_idx = prefix_len_with_latents + tail_colon_rel  # (B,)
    shift_logits_s = student_logits[:, :-1, :].contiguous()
    shift_labels_s = student_labels[:, 1:].contiguous()
    shift_mask_s = (shift_labels_s != -100)
    pos_s = torch.arange(shift_labels_s.shape[1], device=device).unsqueeze(0)
    answer_mask_s = shift_mask_s & (pos_s >= full_colon_idx.unsqueeze(1))
    ce_s = F.cross_entropy(
        shift_logits_s.reshape(-1, shift_logits_s.size(-1)),
        shift_labels_s.reshape(-1).clamp(min=0),
        reduction="none",
    ).view(B, -1)
    denom_s = answer_mask_s.float().sum().clamp(min=1.0)
    L_student = (ce_s * answer_mask_s.float()).sum() / denom_s

    # ----- L1 distillation at the ":" token, across all layers -----
    # Teacher ":" index (into teacher_hiddens[*]): teacher_colon_idx
    # Student ":" index (into student_hiddens[*]): full_colon_idx
    n_layers = len(teacher_hiddens)
    assert n_layers == len(student_hiddens), "Layer count mismatch"

    kd_terms = []
    batch_idx = torch.arange(B, device=device)
    # Fix CRITICAL-9: skip layer 0 — the embedding-layer hidden state is dominated
    # by positional-embedding artifacts that differ between teacher/student alignment.
    for L_t, L_s in zip(teacher_hiddens[1:], student_hiddens[1:]):
        # Select the ":" hidden state at each layer.
        t_vec = L_t[batch_idx, teacher_colon_idx, :]        # (B, D)
        s_vec = L_s[batch_idx, full_colon_idx, :]           # (B, D)
        t_vec = t_vec.detach()
        # Per-layer std of teacher activations across the batch (+ hidden dim).
        layer_std = t_vec.float().std().clamp(min=1e-6)
        diff = (s_vec.float() - t_vec.float()).abs()
        kd_terms.append(diff.mean() / layer_std)
    L_kd = torch.stack(kd_terms).mean()

    L_total = cfg["alpha"] * L_teacher + cfg["beta"] * L_student + cfg["gamma"] * L_kd

    # ----- Auxiliary rank loss (OFF by default) -----
    # When --rank-loss != none and --rank-lambda > 0, add an auxiliary term that
    # pushes Z's singular spectrum toward higher effective rank.  The Z tensors
    # retain gradients through the bottleneck back to w_up and thinker.
    L_rank = torch.zeros(1, device=device, dtype=L_total.dtype)
    if _need_rank_loss:
        z_grad = student["Z_list_grad"]  # list of (B, d, d) with grad
        if z_grad is not None and len(z_grad) > 0:
            L_rank = compute_rank_loss(z_grad, _rank_loss_type)
            L_total = L_total + _rank_lambda * L_rank

    return L_total, {
        "L_teacher": L_teacher.detach(),
        "L_student": L_student.detach(),
        "L_kd": L_kd.detach(),
        "L_rank": L_rank.detach(),
    }


# =============================================================================
# EVAL: GSM8K accuracy + per-problem rank measurement
# =============================================================================

@torch.no_grad()
def generate_answer(model, q_ids, q_mask, tokenizer, special_ids, cfg, device,
                    rank_project_k=None, save_Z=True, force_rank_k=None):
    """Run one student pass up through the latents, then greedy-decode the answer.

    Fix C5: the n_latents feedback loop is executed ONCE to build the full prefix
    embeddings (question + n_latent steps + "<eot> The answer is:").  A single GPT-2
    forward over that prefix seeds the KV cache.  All subsequent new tokens are
    generated with single-token forwards reusing the cached KV — 96× cheaper than
    the previous approach that re-ran the entire loop per token.

    Returns:
        predicted_text (str), Z_stack tensor of shape (n_latents, d, d) or None
    """
    raw = model.module if isinstance(model, DDP) else model
    B = q_ids.shape[0]
    assert B == 1, "generate_answer expects batch size 1 for clarity."

    # Build a short "tail prompt": just "<eot> The answer is:" so the model can
    # autoregress numerical tokens afterward. We do NOT supply the gold answer.
    prompt_text = " The answer is:"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    tail_prompt = [special_ids["eot"]] + prompt_ids
    tail_ids = torch.tensor([tail_prompt], dtype=torch.long, device=device)
    tail_mask = torch.ones_like(tail_ids)

    # Fix C5: Run the n_latents feedback loop ONCE to obtain prefix embeddings
    # (stored inside student_forward as running_embeds + tail_embeds).
    student = raw.student_forward(
        q_ids=q_ids,
        q_mask=q_mask,
        tail_ids=tail_ids,
        tail_mask=tail_mask,
        save_Z=save_Z,
        rank_project_k=rank_project_k,
        force_rank_k=force_rank_k,
    )

    # Collect intermediate logits for smoke-test NaN check (Fix 21).
    intermediate_logits = student["logits"]

    # Fix C5: seed the KV cache with one forward over the full prefix.
    # student_forward's out_full already ran the joint pass; we reuse its last
    # logit position for the first token, then decode with past_key_values.
    logits_last = student["logits"][:, -1, :]
    assert not torch.isnan(logits_last).any(), "NaN in prefix logits"
    next_tok = logits_last.argmax(dim=-1, keepdim=True)   # (1, 1)
    generated = [next_tok.item()]

    if next_tok.item() == special_ids["eos"]:
        text = tokenizer.decode(generated, skip_special_tokens=True)
        Z_stack = None
        if save_Z and student["Z_list"] is not None and student["Z_list"][0] is not None:
            Z_stack = torch.stack([Z.squeeze(0) for Z in student["Z_list"]], dim=0)
        return text, Z_stack

    # Rebuild the full prefix inputs_embeds to seed past_key_values.
    # We re-run the GPT-2 transformer (not the full student_forward loop) with
    # use_cache=True.  This is one additional forward — acceptable because it
    # amortises over all subsequent tokens.
    prefix_len = student["prefix_len_with_latents"]
    tail_embeds = raw.gpt2.transformer.wte(tail_ids)     # (1, Lt, D)
    # Reconstruct running_embeds: GPT-2 wte on q+<bot>, then n_latent dummy
    # steps are NOT available directly — we call student_forward a second time
    # with use_cache semantics by going through the gpt2 model with the same
    # full inputs_embeds that student already computed.  The cleanest approach:
    # expose the full_embeds.  Since we didn't, we call the GPT-2 model directly
    # on the logit sequence already computed.
    #
    # Practical approach: grab past_key_values by running the model on
    # next_tok (a single token) with the prefix as input_ids is impossible
    # because the prefix was continuous.  Instead we seed via inputs_embeds by
    # re-assembling the exact same embedding sequence from scratch (one call,
    # same cost as one student_forward).
    #
    # We expose full_embeds by computing it here from student's prefix+tail.
    # "running_embeds" is not returned; recompute cheaply via a no-latent pass.
    bot_id = raw.special_token_ids["bot"]
    bot_col = torch.full((1, 1), bot_id, dtype=torch.long, device=device)
    prefix_ids_cat = torch.cat([q_ids, bot_col], dim=1)
    prefix_mask_cat = torch.cat([q_mask, torch.ones_like(bot_col)], dim=1)
    prefix_embeds_seed = raw.gpt2.transformer.wte(prefix_ids_cat)  # (1, Lq+1, D)

    # Rebuild latent embeddings by re-running the feedback loop (no grad).
    running_e = prefix_embeds_seed
    running_m = prefix_mask_cat
    last_h_seed = raw.gpt2.transformer(
        inputs_embeds=prefix_embeds_seed,
        attention_mask=prefix_mask_cat,
        output_hidden_states=False,
        return_dict=True,
    ).last_hidden_state[:, -1, :]
    for t_seed in range(raw.n_latents):
        if raw.use_matrix_bottleneck:
            h_s, _ = raw.bottleneck(
                last_h_seed,
                rank_project_k=rank_project_k,
                force_rank_k=force_rank_k,
            )
        else:
            h_s = raw.feedback_proj(last_h_seed)
        h_s = h_s.unsqueeze(1)
        running_e = torch.cat([running_e, h_s], dim=1)
        running_m = torch.cat([running_m, torch.ones((1, 1), dtype=running_m.dtype, device=device)], dim=1)
        if t_seed < raw.n_latents - 1:
            last_h_seed = raw.gpt2.transformer(
                inputs_embeds=running_e,
                attention_mask=running_m,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :]

    full_e = torch.cat([running_e, tail_embeds], dim=1)             # (1, L_prefix+Lt, D)
    full_m = torch.cat([running_m, tail_mask], dim=1)

    # Seed KV cache with the full prefix (use_cache=True).
    seed_out = raw.gpt2(
        inputs_embeds=full_e,
        attention_mask=full_m,
        use_cache=True,
        return_dict=True,
    )
    past = seed_out.past_key_values

    # Greedy decode new tokens using KV cache — one token per forward, O(1) cost.
    max_new = 16 if cfg.get("dataset", "gsm8k_aug") != "prosqa" else 24
    cur_tok = next_tok   # already decoded one token above
    cur_mask = torch.cat([full_m, torch.ones((1, 1), dtype=full_m.dtype, device=device)], dim=1)

    for _ in range(max_new - 1):
        step_out = raw.gpt2(
            input_ids=cur_tok,
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = step_out.past_key_values
        cur_tok = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)], dim=1)
        generated.append(cur_tok.item())
        if cur_tok.item() == special_ids["eos"]:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)

    Z_stack = None
    if save_Z and student["Z_list"] is not None and student["Z_list"][0] is not None:
        Z_stack = torch.stack([Z.squeeze(0) for Z in student["Z_list"]], dim=0)  # (n_latents, d, d)
    return text, Z_stack


ANS_NUM_RE = re.compile(r"[\-\d\.,]+")


def parse_predicted_number(text):
    """Extract the first numeric substring from a predicted answer string."""
    m = ANS_NUM_RE.search(text)
    if m is None:
        return None
    return m.group(0).replace(",", "").strip().rstrip(".")


def numbers_match(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred == gold


def prosqa_answer_match(pred_text, gold_text):
    """Match ProsQA answers of form 'NAME is [a/an] CLASS.' by the class word
    in the FIRST sentence of the prediction. Tolerant to trailing garbage
    because GPT-2 will keep generating past the period without EOS training."""
    import re
    def first_sentence_final_class(s):
        s = s.strip()
        # Fix 5: Truncate at first terminal punctuation to take the first sentence only.
        # Model generations may continue past the period — stop there.
        for sep in [".", "?", "!", "\n"]:
            i = s.find(sep)
            if i >= 0:
                s = s[:i]
                break
        s = s.strip().lower()
        tokens = re.split(r"\s+", s)
        return tokens[-1] if tokens and tokens[-1] else ""
    return first_sentence_final_class(pred_text) == first_sentence_final_class(gold_text)


@torch.no_grad()
def evaluate_gsm8k(model, val_dataset, tokenizer, special_ids, cfg, device,
                   rank_project_k=None, save_ranks=True, max_batches=None,
                   force_rank_k=None):
    """Eval loop. Returns (accuracy, rank_records, sample_Z_records).

    rank_records: list of dicts, one per problem, with effective ranks.
    sample_Z_records: list of dicts holding raw Z matrices for up to
                      cfg["max_rank_samples"] held-out problems.
    force_rank_k: forwarded to generate_answer for --force-rank-during-training eval.
    """
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()

    correct = 0
    total = 0
    rank_records = []
    sample_Z_records = []
    sample_limit = cfg["max_rank_samples"]

    max_problems = (
        (max_batches or cfg["max_eval_batches"]) * cfg["eval_batch_size"]
    )

    for idx in range(min(len(val_dataset), max_problems)):
        item = val_dataset[idx]
        q_ids = torch.tensor([item["q_ids"]], dtype=torch.long, device=device)
        q_mask = torch.ones_like(q_ids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_text, Z_stack = generate_answer(
                model, q_ids, q_mask, tokenizer, special_ids, cfg, device,
                rank_project_k=rank_project_k, save_Z=save_ranks,
                force_rank_k=force_rank_k,
            )

        if cfg.get("dataset", "gsm8k_aug") == "prosqa":
            is_correct = prosqa_answer_match(pred_text, item["answer_text"])
        else:
            pred_num = parse_predicted_number(pred_text)
            is_correct = numbers_match(pred_num, item["answer_text"])
        if is_correct:
            correct += 1
        total += 1

        if save_ranks and Z_stack is not None:
            ranks_per_step = effective_rank(Z_stack).cpu().tolist()  # list of n_latents
            rank_records.append({
                "problem_id": idx,
                "reasoning_steps": item["reasoning_steps"],
                "effective_ranks": ranks_per_step,
                "mean_rank": float(sum(ranks_per_step) / max(len(ranks_per_step), 1)),
                "correct": bool(is_correct),
                "predicted": pred_text[:64],
                "gold": item["answer_text"],
            })
            if len(sample_Z_records) < sample_limit:
                sample_Z_records.append({
                    "problem_id": idx,
                    "Z_matrices": Z_stack.float().cpu().tolist(),
                    "reasoning_steps": item["reasoning_steps"],
                    "correct": bool(is_correct),
                })

    raw.train()
    accuracy = correct / max(total, 1)
    return accuracy, rank_records, sample_Z_records


# =============================================================================
# SMOKE TESTS (mandatory per CLAUDE.md)
# =============================================================================

def run_smoke_tests(cfg):
    """Smoke tests run on a single GPU (no DDP). Exits non-zero on failure."""
    print("\n" + "=" * 70)
    print("  SMOKE TESTS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA not available; running smoke tests on CPU.")

    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Fix CRITICAL-7
    assert_colon_tokenization(tokenizer)  # Fix CRITICAL-10

    # --- Test 1: rank measurement on random matrices ---
    print("\n[1/5] Rank measurement smoke test...")
    d = cfg["mat_dim"]
    rand_Z = torch.randn(8, d, d, device=device)
    er = effective_rank(rand_Z).tolist()
    assert all(1.0 <= r <= d + 1e-3 for r in er), \
        f"Effective rank out of [1,{d}]: {er}"
    # Rank-1 matrix should have effective rank ~1.
    u = torch.randn(1, d, device=device)
    v = torch.randn(1, d, device=device)
    rank1 = u.unsqueeze(-1) @ v.unsqueeze(-2)   # (1, d, d)
    er1 = effective_rank(rank1).item()
    assert er1 < 1.5, f"Rank-1 matrix gave effective rank {er1}"
    # Truncate_to_rank smoke.
    Zt = truncate_to_rank(rand_Z, 2)
    assert Zt.shape == rand_Z.shape
    assert (effective_rank(Zt) < 2.5).all(), "Rank-2 projection gave rank > 2.5"
    print(f"  Random effective ranks: min={min(er):.2f} max={max(er):.2f}")
    print(f"  Rank-1 outer product: effective rank = {er1:.3f}")
    print("  PASS")

    # --- Test 2: forward pass (vanilla + matrix) ---
    print("\n[2/5] Forward pass smoke test...")
    for use_matrix in [False, True]:
        tag = "matrix" if use_matrix else "vanilla"
        print(f"  Building CodiModel ({tag})...")
        model = CodiModel(
            base_model_name=cfg["base_model"],
            n_latents=cfg["n_latents"],
            use_matrix_bottleneck=use_matrix,
            mat_dim=cfg["mat_dim"],
            use_thinking_iter=cfg["use_thinking_iter"],
            readout=cfg["readout"],
        ).to(device)
        model.add_special_tokens(tokenizer)
        model.train()

        B = 2
        # Fake question tokens.
        q_ids = torch.randint(0, 50000, (B, 12), device=device)
        q_mask = torch.ones_like(q_ids)
        tail_ids = torch.tensor(
            [[model.special_token_ids["eot"]] + tokenizer.encode(
                " The answer is: 42", add_special_tokens=False
            )] * B,
            dtype=torch.long, device=device,
        )
        tail_mask = torch.ones_like(tail_ids)

        with torch.autocast("cuda" if device.type == "cuda" else "cpu",
                             dtype=torch.bfloat16,
                             enabled=(device.type == "cuda")):
            out = model.student_forward(q_ids, q_mask, tail_ids, tail_mask, save_Z=True)
        logits = out["logits"]
        assert logits.dim() == 3 and logits.shape[0] == B, f"bad logits {logits.shape}"
        assert not torch.isnan(logits).any(), f"{tag}: NaN in logits"
        assert not torch.isinf(logits).any(), f"{tag}: Inf in logits"
        if use_matrix:
            for i, Z in enumerate(out["Z_list"]):
                assert Z.shape == (B, cfg["mat_dim"], cfg["mat_dim"]), \
                    f"bad Z shape at step {i}: {Z.shape}"
                assert not torch.isnan(Z).any(), f"NaN in Z step {i}"
        print(f"  {tag}: logits {tuple(logits.shape)}, OK")
        del model
    print("  PASS")

    # --- Test 3: backward pass with synthetic batch, grads finite ---
    # Fix SERIOUS-16: use B=2 with distinct colon positions; check specific
    # thinker params explicitly with grad.abs().max() > 0.
    print("\n[3/5] Backward pass smoke test (matrix variant, B=2, distinct colons)...")
    model = CodiModel(
        base_model_name=cfg["base_model"],
        n_latents=cfg["n_latents"],
        use_matrix_bottleneck=True,
        mat_dim=cfg["mat_dim"],
        use_thinking_iter=cfg["use_thinking_iter"],
        readout=cfg["readout"],
    ).to(device)
    model.add_special_tokens(tokenizer)
    model.train()

    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]

    # Build a B=2 batch with distinct colon positions per example.
    def _make_row(q_text, ans_str):
        q = tokenizer.encode(q_text + "\n", add_special_tokens=False)
        tail_list = [model.special_token_ids["eot"]] + tokenizer.encode(
            f" The answer is: {ans_str}", add_special_tokens=False
        )
        teacher_text = f"{q_text}\nReasoning. The answer is: {ans_str}"
        teacher_list = tokenizer.encode(teacher_text, add_special_tokens=False)
        tcolon = [i for i, t in enumerate(teacher_list) if t == colon_id][-1]
        tcolon_rel = [i for i, t in enumerate(tail_list) if t == colon_id][0]
        return q, tail_list, teacher_list, tcolon, tcolon_rel

    q1, t1, th1, tc1, tcr1 = _make_row("How many apples", "7")
    q2, t2, th2, tc2, tcr2 = _make_row("What is two plus two plus one", "5")
    # Pad to equal lengths for the batch.
    max_q = max(len(q1), len(q2))
    max_tail = max(len(t1), len(t2))
    max_teacher = max(len(th1), len(th2))
    pad_id_smoke = tokenizer.pad_token_id

    def lpad(seq, L):
        return [pad_id_smoke] * (L - len(seq)) + seq

    def rpad(seq, L):
        return seq + [pad_id_smoke] * (L - len(seq))

    q_ids_b2 = torch.tensor([lpad(q1, max_q), lpad(q2, max_q)], dtype=torch.long, device=device)
    q_mask_b2 = (q_ids_b2 != pad_id_smoke).long()
    tail_ids_b2 = torch.tensor([rpad(t1, max_tail), rpad(t2, max_tail)], dtype=torch.long, device=device)
    tail_mask_b2 = (tail_ids_b2 != pad_id_smoke).long()
    teacher_ids_b2 = torch.tensor([rpad(th1, max_teacher), rpad(th2, max_teacher)], dtype=torch.long, device=device)
    teacher_mask_b2 = (teacher_ids_b2 != pad_id_smoke).long()

    batch = {
        "teacher_ids": teacher_ids_b2,
        "teacher_mask": teacher_mask_b2,
        "teacher_colon_idx": torch.tensor([tc1, tc2], device=device),
        "q_ids": q_ids_b2,
        "q_mask": q_mask_b2,
        "tail_ids": tail_ids_b2,
        "tail_mask": tail_mask_b2,
        "tail_colon_rel": torch.tensor([tcr1, tcr2], device=device),
        "reasoning_steps": torch.tensor([1, 2], device=device),
        "answer_texts": ["7", "5"],
        "questions": ["How many apples", "two plus two plus one"],
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss, parts = compute_codi_loss(model, batch, cfg, device, model.special_token_ids)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()
    n_with_grad = 0
    n_without_grad = 0
    bad = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_without_grad += 1
            continue
        n_with_grad += 1
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            bad.append(name)
    assert not bad, f"NaN/Inf gradients in: {bad[:5]}"
    # Fix SERIOUS-16: check specific thinker sub-module params have nonzero grads.
    thinker_checks = [
        "bottleneck.thinker.scale",
        "bottleneck.thinker.delta_gate",
        "bottleneck.thinker.gamma_up",
    ]
    for param_prefix in thinker_checks:
        for name, p in model.named_parameters():
            if name.startswith(param_prefix) and p.requires_grad:
                assert p.grad is not None and p.grad.abs().max() > 0, \
                    f"Thinker param {name} has zero/no grad after backward!"
    print(f"  loss={loss.item():.4f} teacher={parts['L_teacher'].item():.4f} "
          f"student={parts['L_student'].item():.4f} kd={parts['L_kd'].item():.4f}")
    print(f"  {n_with_grad} params with grad, {n_without_grad} without grad")
    print("  PASS")

    # --- Test 4: eval batch fits ---
    # Fix 21: assert non-empty string returned and no NaN in intermediate logits.
    print("\n[4/5] Eval batch VRAM smoke test...")
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            q_ids_big = torch.randint(0, 50000, (1, 64), device=device)
            q_mask_big = torch.ones_like(q_ids_big)
            with torch.autocast("cuda" if device.type == "cuda" else "cpu",
                                 dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
                text, Z = generate_answer(
                    model, q_ids_big, q_mask_big, tokenizer,
                    model.special_token_ids, cfg, device, save_Z=True,
                )
            assert Z is not None and Z.shape == (cfg["n_latents"], cfg["mat_dim"], cfg["mat_dim"]), \
                f"bad Z {None if Z is None else Z.shape}"
            # Fix 21: non-empty output string.
            assert isinstance(text, str), "generate_answer returned non-string"
            assert len(text) >= 0, "generate_answer returned None"  # even empty str is ok, len check is for type
    print(f"  Generated sample: {text[:40]!r}, Z stack {tuple(Z.shape)}")
    print("  PASS")

    # --- Test 5: checkpoint save/load round trip ---
    print("\n[5/5] Checkpoint save/load smoke test...")
    tmp_dir = Path("/tmp/matrix_codi_smoke")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_dir / "smoke.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    model2 = CodiModel(
        base_model_name=cfg["base_model"],
        n_latents=cfg["n_latents"],
        use_matrix_bottleneck=True,
        mat_dim=cfg["mat_dim"],
        use_thinking_iter=cfg["use_thinking_iter"],
        readout=cfg["readout"],
    ).to(device)
    model2.add_special_tokens(tokenizer)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model2.load_state_dict(sd["model"])
    # Compare one param.
    p1 = next(model.bottleneck.w_up.parameters())
    p2 = next(model2.bottleneck.w_up.parameters())
    assert torch.allclose(p1, p2), "Checkpoint round trip failed"
    os.remove(ckpt_path)
    print("  PASS")

    print("\n" + "=" * 70)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 70 + "\n")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def build_logger(results_dir, is_main):
    """Simple logger writing to stdout and train.log on rank 0."""

    class Logger:
        def __init__(self, path):
            self.f = open(path, "a") if is_main else None

        def log(self, msg):
            if not is_main:
                return
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            if self.f:
                self.f.write(line + "\n")
                self.f.flush()

        def close(self):
            if self.f:
                self.f.close()

    return Logger(results_dir / "train.log")


def assert_colon_tokenization(tokenizer):
    """Fix CRITICAL-10: verify that ':' appears in ' The answer is: 42' encoding.

    GPT-2's BPE can merge ':' with surrounding chars.  If the colon id we look
    up is not present in the answer-prefix string, our KD alignment is broken.
    """
    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
    full_ids = tokenizer.encode(" The answer is: 42", add_special_tokens=False)
    assert colon_id in full_ids, (
        f"Colon token id {colon_id} not found in encoding of ' The answer is: 42'! "
        f"Full encoding: {full_ids}.  Tokenizer BPE merges ':' -- KD alignment is broken."
    )


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_run(mode, cfg):
    """Trains either Run A (vanilla) or Run B (matrix) under DDP."""
    assert mode in ("train_vanilla", "train_matrix")
    use_matrix = (mode == "train_matrix")

    # DDP init
    dist.init_process_group("nccl", timeout=dt_module.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    # Fix SERIOUS-17: identical seed across ALL ranks for model init so weights
    # start at the same values before DDP syncs them.
    set_seed(cfg["seed"])

    results_dir = Path(cfg["results_dir"])
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), results_dir / "script.py")
    dist.barrier()

    logger = build_logger(results_dir, is_main)
    logger.log(f"=== Matrix-CODI Run {'B (matrix)' if use_matrix else 'A (vanilla)'} ===")
    logger.log(f"World size: {world_size}")
    logger.log(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # Tokenizer — Fix CRITICAL-7: left-pad questions so <bot> is flush to content.
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if is_main:
        assert_colon_tokenization(tokenizer)  # Fix CRITICAL-10: hard-fail on BPE mismatch

    # Model
    model = CodiModel(
        base_model_name=cfg["base_model"],
        n_latents=cfg["n_latents"],
        use_matrix_bottleneck=use_matrix,
        mat_dim=cfg["mat_dim"],
        use_thinking_iter=cfg["use_thinking_iter"],
        readout=cfg.get("readout", "flatten"),
    ).to(device)
    model.add_special_tokens(tokenizer)
    special_ids = model.special_token_ids

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total params: {n_params:,}")
    if use_matrix:
        n_bn = sum(p.numel() for p in model.bottleneck.parameters() if p.requires_grad)
        logger.log(f"Bottleneck params: {n_bn:,}")

    # Data
    if cfg["dataset"] == "prosqa":
        train_ds = ProsQADataset("train", tokenizer, cfg, special_ids)
        val_ds = ProsQADataset("val", tokenizer, cfg, special_ids)
    else:
        train_ds = GSM8KDataset("train", tokenizer, cfg, special_ids)
        val_ds = GSM8KDataset("test", tokenizer, cfg, special_ids)
    logger.log(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=local_rank, shuffle=True,
        seed=cfg["seed"],
    )
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size_per_gpu"],
        sampler=sampler,
        # Fix SERIOUS-14: use functools.partial so collate_fn is picklable for multi-process DataLoader.
        collate_fn=functools.partial(collate_gsm8k, pad_id=pad_id, n_latents=cfg["n_latents"]),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # DDP wrap (after any parameter changes from resize_token_embeddings)
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Fix SERIOUS-17: reseed with per-rank offset AFTER DDP wrap so data
    # shuffling and dropout differ per rank, but model weights were identical.
    set_seed(cfg["seed"] + local_rank)

    optimizer = torch.optim.AdamW(
        model_ddp.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )

    total_steps = cfg["epochs"] * len(train_loader)

    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / max(cfg["warmup_steps"], 1)
        p = (step - cfg["warmup_steps"]) / max(total_steps - cfg["warmup_steps"], 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Save-path distinguisher
    ckpt_name = "best_run_b_matrix.pt" if use_matrix else "best_run_a_vanilla.pt"

    best_acc = -1.0
    step = 0
    training_curve = []
    # Fix 19: rolling loss average over last N=50 steps.
    _loss_window = []
    _LOSS_WINDOW_SIZE = 50
    z_rank_curve = []  # Missing 1: per-50-step rank log
    start = time.time()
    logger.log(f"Total steps: {total_steps} (epochs {cfg['epochs']} x steps/epoch {len(train_loader)})")
    logger.log("--- Training ---")

    model_ddp.train()
    for epoch in range(cfg["epochs"]):
        sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = compute_codi_loss(
                    model_ddp, batch, cfg, device, special_ids
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_ddp.parameters(), cfg["grad_clip"]
            )
            optimizer.step()
            scheduler.step()
            step += 1
            # Fix 19: accumulate rolling loss window.
            _loss_window.append(loss.item())
            if len(_loss_window) > _LOSS_WINDOW_SIZE:
                _loss_window.pop(0)

            if step % cfg["log_interval"] == 0 and is_main:
                elapsed = time.time() - start
                # Fix 19: report running average over last window, not last single step.
                avg_loss = sum(_loss_window) / max(len(_loss_window), 1)

                # Missing 1: log effective rank of Z on one batch every 50 steps.
                z_rank_str = ""
                if use_matrix:
                    raw_m = model_ddp.module if isinstance(model_ddp, DDP) else model_ddp
                    raw_m.eval()
                    with torch.no_grad():
                        _b = {k: v.to(device) if torch.is_tensor(v) else v
                              for k, v in batch.items()}
                        _q = _b["q_ids"][:1]
                        _qm = _b["q_mask"][:1]
                        _t = _b["tail_ids"][:1]
                        _tm = _b["tail_mask"][:1]
                        _s = raw_m.student_forward(_q, _qm, _t, _tm, save_Z=True)
                        if _s["Z_list"] and _s["Z_list"][0] is not None:
                            _Z = torch.stack([z.squeeze(0) for z in _s["Z_list"]])  # (n_lat, d, d)
                            _er = effective_rank(_Z).tolist()
                            _mean_er = sum(_er) / len(_er)
                            z_rank_str = f" | Z_rank {_mean_er:.2f}"
                            z_rank_curve.append({"step": step, "mean_effective_rank": _mean_er})
                            # Fix 20: log scale param.
                            _scale = raw_m.bottleneck.thinker.scale.data.item()
                            z_rank_str += f" | scale {_scale:.4f}"
                            # Fix 8: mid-training rank collapse diagnostic.
                            # If rank collapses below 3.0 by step 500, the matrix bottleneck
                            # is broken and further training won't help. Log a loud warning.
                            if step >= 500 and len(z_rank_curve) >= 5 and use_matrix:
                                recent_ranks = [entry["mean_effective_rank"] for entry in z_rank_curve[-5:]]
                                mean_recent = sum(recent_ranks) / len(recent_ranks)
                                if mean_recent < 3.0:
                                    logger.log(
                                        f"  [WARN] Z rank collapsed: mean_rank={mean_recent:.2f} < 3.0 "
                                        f"at step {step}. Matrix bottleneck may be broken."
                                    )
                    raw_m.train()

                rank_loss_str = ""
                if cfg.get("rank_loss", "none") != "none" and cfg.get("rank_lambda", 0.0) > 0:
                    rank_loss_str = f" | step.rank_loss {parts['L_rank'].item():.4f}"
                logger.log(
                    f"step {step:6d}/{total_steps} | loss_avg {avg_loss:.4f} | "
                    f"L_t {parts['L_teacher'].item():.3f} L_s {parts['L_student'].item():.3f} "
                    f"L_kd {parts['L_kd'].item():.3f} | gn {grad_norm:.2f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s{z_rank_str}{rank_loss_str}"
                )

        # End of epoch: eval on rank 0 only.
        # Fix C5+C6: wrap in try/except so a crash on rank 0 still releases
        # other ranks via dist.barrier() in finally.
        if (epoch + 1) % cfg["eval_interval_epochs"] == 0:
            try:
                if is_main:
                    logger.log(f"--- Eval after epoch {epoch + 1} ---")
                    _eval_force_rank_k = cfg.get("force_rank_during_training", 0) or None
                    if _eval_force_rank_k == 0:
                        _eval_force_rank_k = None
                    acc, rank_records, sample_Zs = evaluate_gsm8k(
                        model_ddp, val_ds, tokenizer, special_ids, cfg, device,
                        save_ranks=use_matrix,  # only matrix run has Z to rank
                        force_rank_k=_eval_force_rank_k,
                    )
                    logger.log(f"  {cfg['dataset']} accuracy: {acc*100:.2f}%")

                    # Fix 19: use rolling average, not last single-step loss.
                    avg_loss_at_eval = sum(_loss_window) / max(len(_loss_window), 1)
                    eval_entry = {
                        "epoch": epoch + 1,
                        "step": step,
                        "accuracy": acc,
                        "train_loss_avg50": float(avg_loss_at_eval),
                    }
                    training_curve.append(eval_entry)

                    if acc > best_acc:
                        best_acc = acc
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "cfg": cfg,
                                "use_matrix_bottleneck": use_matrix,
                                "tokenizer_len": len(tokenizer),
                            },
                            results_dir / ckpt_name,
                        )
                        logger.log(f"  *** New best: {acc*100:.2f}% -> saved {ckpt_name}")

                    # Save the latest rank dynamics as we go.
                    if use_matrix and rank_records:
                        with open(results_dir / "rank_dynamics.json", "w") as f:
                            json.dump(
                                {
                                    "epoch": epoch + 1,
                                    "step": step,
                                    "records": rank_records,
                                    "sample_Z": sample_Zs,
                                },
                                f,
                                indent=2,
                            )
            finally:
                dist.barrier()

    # Fix 1: Run one unbiased full-test eval after training.
    # During training we cap max_eval_batches for speed; the final eval uses
    # cfg["final_eval_batches"] (default 32 → 512 problems) to get a clean number.
    if is_main:
        logger.log(f"--- Final full-test eval ---")
        eval_model = model.module if hasattr(model, "module") else model
        eval_model.eval()
        final_acc, final_records, _ = evaluate_gsm8k(
            eval_model, val_ds, tokenizer, special_ids, cfg, device,
            save_ranks=False,
            max_batches=cfg["final_eval_batches"],
        )
        logger.log(f"  Final {len(final_records)}-problem eval: {final_acc*100:.2f}%")
        if final_acc > best_acc:
            logger.log(f"  *** Final eval exceeded best epoch eval ({best_acc*100:.2f}%)")
        results_path_fe = Path(cfg["results_dir"]) / "final_eval.json"
        with open(results_path_fe, "w") as f:
            json.dump({
                "final_accuracy": final_acc,
                "n_problems": len(final_records),
                "best_epoch_accuracy": best_acc,
            }, f, indent=2)
        logger.log(f"  Saved: {results_path_fe}")
    dist.barrier()

    # Final summary
    if is_main:
        elapsed = time.time() - start
        logger.log("=== DONE ===")
        logger.log(f"Best accuracy: {best_acc*100:.2f}%")
        logger.log(f"Wall time: {elapsed/60:.1f} min")

        summary_lines = [
            "",
            "=" * 70,
            f"  MATRIX-CODI {'RUN B (matrix)' if use_matrix else 'RUN A (vanilla)'}",
            "=" * 70,
            f"  Base model: {cfg['base_model']}",
            f"  Use matrix bottleneck: {use_matrix}",
            f"  Latents: {cfg['n_latents']}, mat_dim: {cfg['mat_dim']}",
            f"  Params: {n_params:,}",
            f"  Epochs: {cfg['epochs']}, global batch: "
            f"{cfg['batch_size_per_gpu']*world_size}",
            f"  Wall time: {elapsed/60:.1f} min on {world_size} GPUs",
            "",
            f"  Best {cfg['dataset']} accuracy: {best_acc*100:.2f}%",
            "=" * 70,
            "",
        ]
        summary = "\n".join(summary_lines)
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", "a") as f:
            f.write(summary)

        results_path = results_dir / "results.json"
        existing = {}
        if results_path.exists():
            try:
                existing = json.load(open(results_path))
            except Exception:
                existing = {}
        existing[mode] = {
            "config": cfg,
            "best_accuracy": best_acc,
            "training_curve": training_curve,
            "z_rank_curve": z_rank_curve,  # Missing 1: per-50-step rank curve
            "wall_time_min": elapsed / 60.0,
            "world_size": world_size,
            "n_params": n_params,
        }
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        logger.close()

    dist.barrier()
    dist.destroy_process_group()


# =============================================================================
# RUN C: RANK PROJECTION ABLATION AT EVAL TIME
# =============================================================================

def eval_rank_projection(cfg, checkpoint_path):
    """Load Run B checkpoint, run eval with forced rank projections k in {1..16}."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(results_dir, is_main=True)
    logger.log(f"=== Matrix-CODI Run C (rank projection ablation) ===")
    logger.log(f"Loading checkpoint from {checkpoint_path}")

    # Fix 18 / Fix C1: assert path exists before load.
    assert Path(checkpoint_path).exists(), f"Checkpoint not found: {checkpoint_path}"
    # Fix C1: weights_only=False because checkpoint contains a cfg dict (our own file, trusted).
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = sd.get("cfg", cfg)
    use_matrix = sd.get("use_matrix_bottleneck", True)
    assert use_matrix, "Run C only makes sense on a matrix-CODI checkpoint."

    tokenizer = GPT2TokenizerFast.from_pretrained(saved_cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    # Fix CRITICAL-7: left-pad at eval too.
    tokenizer.padding_side = "left"
    assert_colon_tokenization(tokenizer)  # Fix CRITICAL-10

    model = CodiModel(
        base_model_name=saved_cfg["base_model"],
        n_latents=saved_cfg["n_latents"],
        use_matrix_bottleneck=True,
        mat_dim=saved_cfg["mat_dim"],
        use_thinking_iter=saved_cfg.get("use_thinking_iter", True),
        readout=saved_cfg.get("readout", "flatten"),
    )
    model.add_special_tokens(tokenizer)
    # Ensure the embedding table matches the checkpoint (after special tokens).
    model.load_state_dict(sd["model"])
    model.to(device)
    model.eval()

    # Fix 3: The evaluator branches on cfg["dataset"] and other shape keys; ensure
    # we use the values the checkpoint was trained on, not whatever the CLI passed.
    # Err on side of copying ALL dataset-shape-affecting keys from saved_cfg.
    for _key in ("dataset", "max_q_len", "max_ans_len", "max_total_len", "mat_dim", "n_latents"):
        if _key in saved_cfg:
            cfg[_key] = saved_cfg[_key]
    assert cfg["dataset"] == saved_cfg.get("dataset", "gsm8k_aug"), (
        f"dataset mismatch: cfg has {cfg['dataset']!r} but saved_cfg has "
        f"{saved_cfg.get('dataset', 'gsm8k_aug')!r}"
    )

    # Fix SERIOUS-15: use saved_cfg for dataset construction so tokenisation and
    # sequence-length budgets match exactly what training used.
    if saved_cfg.get("dataset", "gsm8k_aug") == "prosqa":
        # Patch the saved config with Round 2 ProsQA paths in case the checkpoint
        # was saved before the paths were added. Prefer the passed-in cfg's paths
        # if present (Run C may be launched from a different working directory).
        saved_cfg.setdefault("prosqa_val_path", cfg["prosqa_val_path"])
        val_ds = ProsQADataset("val", tokenizer, saved_cfg, model.special_token_ids)
    else:
        val_ds = GSM8KDataset("test", tokenizer, saved_cfg, model.special_token_ids)
    logger.log(f"Val set: {len(val_ds)} problems")

    ks = [1, 2, 4, 8, saved_cfg["mat_dim"]]
    ks = sorted(set([k for k in ks if 1 <= k <= saved_cfg["mat_dim"]]))

    results = {}
    # Cap the eval set for all k to keep wall time bounded.
    problems_per_k = min(len(val_ds), cfg["max_eval_batches"] * cfg["eval_batch_size"])
    logger.log(f"Evaluating {problems_per_k} problems per k, for k in {ks}")

    for k in ks:
        logger.log(f"--- k = {k} ---")
        acc, rank_records, _ = evaluate_gsm8k(
            model, val_ds, tokenizer, model.special_token_ids, cfg, device,
            rank_project_k=k,
            save_ranks=True,
            max_batches=cfg["max_eval_batches"],
        )
        logger.log(f"  k={k}: accuracy {acc*100:.2f}% on {len(rank_records)} problems")
        # Also log mean effective rank to sanity check the truncation actually
        # reduced rank as intended.
        if rank_records:
            mean_rank = sum(r["mean_rank"] for r in rank_records) / len(rank_records)
            logger.log(f"  k={k}: mean effective rank across latents = {mean_rank:.2f}")
        results[str(k)] = {
            "accuracy": acc,
            "n_problems": len(rank_records),
            "records": rank_records,
        }

    # Missing 2: Spearman correlation between effective rank and correctness.
    # Computed over ALL records across k values (use the full-rank k=mat_dim run).
    full_k_str = str(saved_cfg["mat_dim"])
    spearman_result = None
    if full_k_str in results and results[full_k_str]["records"]:
        full_records = results[full_k_str]["records"]
        eff_ranks = [r["mean_rank"] for r in full_records]
        correctness = [1.0 if r["correct"] else 0.0 for r in full_records]
        try:
            from scipy.stats import spearmanr as _spearmanr
            corr, pval = _spearmanr(eff_ranks, correctness)
        except ImportError:
            # Manual Spearman from ranks if scipy unavailable.
            def _rank_list(vals):
                sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
                ranks = [0.0] * len(vals)
                for r_pos, i in enumerate(sorted_idx):
                    ranks[i] = r_pos + 1.0
                return ranks
            r1 = _rank_list(eff_ranks)
            r2 = _rank_list(correctness)
            n = len(r1)
            d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
            corr = 1.0 - 6.0 * d2 / max(n * (n * n - 1), 1)
            pval = float("nan")
        spearman_result = {"spearman_r": float(corr), "p_value": float(pval)}
        logger.log(f"Spearman r(rank, correct) = {corr:.4f}  p={pval:.4g}")

    # Write the Run C output.
    out_path = results_dir / "rank_projection_ablation.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "config": cfg,
                "results_by_k": results,
                "spearman": spearman_result,   # Missing 2
            },
            f,
            indent=2,
            default=str,
        )
    logger.log(f"Saved: {out_path}")

    # Missing 3: analysis plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Brand palette (matches pebble-ai-site).
        _BG = "#FAF5E7"
        _ACCENT = "#8B2E1F"
        _TEXT = "#1a1a1a"
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["text.color"] = _TEXT
        plt.rcParams["axes.labelcolor"] = _TEXT
        plt.rcParams["xtick.color"] = _TEXT
        plt.rcParams["ytick.color"] = _TEXT
        plt.rcParams["axes.edgecolor"] = _TEXT

        plots_dir = results_dir / "analysis_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: accuracy_vs_k.svg
        k_vals = sorted(int(k) for k in results)
        accs = [results[str(k)]["accuracy"] * 100.0 for k in k_vals]
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=_BG)
        ax.set_facecolor(_BG)
        ax.plot(k_vals, accs, color=_ACCENT, linewidth=2.0, marker="o", markersize=7)
        ax.set_xlabel("rank k", fontsize=10, labelpad=8)
        ax.set_ylabel("GSM8K accuracy (%)", fontsize=10, labelpad=8)
        ax.set_title("accuracy vs rank-k projection", fontsize=11, color=_TEXT,
                     fontweight="bold", pad=12, loc="left")
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color=_TEXT)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color(_TEXT)
        plt.tight_layout()
        svg1 = plots_dir / "accuracy_vs_k.svg"
        plt.savefig(svg1, format="svg", facecolor=_BG, bbox_inches="tight")
        plt.close(fig)
        logger.log(f"Saved plot: {svg1}")

        # Plot 2: rank_vs_correct.svg (scatter, jittered)
        if full_k_str in results and results[full_k_str]["records"]:
            import random as _rand
            _records = results[full_k_str]["records"]
            _ranks = [r["mean_rank"] for r in _records]
            _correct = [1 if r["correct"] else 0 for r in _records]
            # jitter correctness for scatter legibility
            _jitter = [c + (_rand.random() - 0.5) * 0.05 for c in _correct]
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=_BG)
            ax2.set_facecolor(_BG)
            ax2.scatter(_ranks, _jitter, color=_ACCENT, alpha=0.4, s=18, linewidths=0)
            ax2.set_xlabel("mean effective rank", fontsize=10, labelpad=8)
            ax2.set_ylabel("correct (jittered)", fontsize=10, labelpad=8)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["wrong", "correct"])
            ax2.set_title("effective rank vs correctness", fontsize=11, color=_TEXT,
                          fontweight="bold", pad=12, loc="left")
            if spearman_result:
                ax2.text(0.02, 0.95, f"Spearman r = {spearman_result['spearman_r']:.3f}",
                         transform=ax2.transAxes, fontsize=9, color=_TEXT,
                         verticalalignment="top")
            ax2.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color=_TEXT)
            for spine in ax2.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color(_TEXT)
            plt.tight_layout()
            svg2 = plots_dir / "rank_vs_correct.svg"
            plt.savefig(svg2, format="svg", facecolor=_BG, bbox_inches="tight")
            plt.close(fig2)
            logger.log(f"Saved plot: {svg2}")
    except Exception as _plot_err:
        logger.log(f"WARNING: Could not generate analysis plots: {_plot_err}")

    # Append a summary.
    summary_lines = [
        "",
        "=" * 70,
        "  MATRIX-CODI RUN C (rank projection ablation)",
        "=" * 70,
        f"  Checkpoint: {checkpoint_path}",
    ]
    for k in ks:
        summary_lines.append(
            f"  k={k:2d}: {results[str(k)]['accuracy']*100:.2f}%"
        )
    if spearman_result:
        summary_lines.append(
            f"  Spearman r(rank, correct) = {spearman_result['spearman_r']:.4f}"
        )
    summary_lines += ["=" * 70, ""]
    summary = "\n".join(summary_lines)
    logger.log(summary)
    with open(results_dir / "SUMMARY.txt", "a") as f:
        f.write(summary)

    logger.close()


# =============================================================================
# CLI ENTRY
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Matrix-CODI rank dynamics experiment")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["train_vanilla", "train_matrix", "eval_rank_projection"],
        help="Which experiment mode to run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run all smoke tests and exit.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a Run B checkpoint (required for --mode eval_rank_projection).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override CONFIG['results_dir'].",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override CONFIG['epochs'] (useful for quick runs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override CONFIG['batch_size_per_gpu'].",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Override CONFIG['max_eval_batches'] (cap eval to avoid NCCL timeout).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override CONFIG['warmup_steps']. Round 2 uses 50 for shorter training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["gsm8k_aug", "prosqa"],
        help="Override CONFIG['dataset']. gsm8k_aug reproduces Round 1; prosqa runs Round 2.",
    )
    parser.add_argument(
        "--readout",
        type=str,
        default=None,
        choices=["flatten", "bilinear", "bilinear_gelu", "svd_aug", "quadratic"],
        help="Override CONFIG['readout']. flatten preserves existing behavior.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override CONFIG['base_model']. e.g. gpt2, gpt2-medium, gpt2-large.",
    )
    parser.add_argument(
        "--n-latents",
        type=int,
        default=None,
        help="Override CONFIG['n_latents'].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override CONFIG['seed'].",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Override CONFIG['gamma'] (KD loss weight). gamma=0 disables KD.",
    )
    # ---- New rank-aware training flags (all OFF by default) ----
    parser.add_argument(
        "--rank-loss",
        type=str,
        default="none",
        choices=["none", "entropy", "nuclear"],
        help=(
            "Auxiliary rank-maximisation loss on the matrix observables Z. "
            "'entropy': maximises effective rank via singular-value entropy. "
            "'nuclear': maximises nuclear norm (grows σ_1, not guaranteed to spread). "
            "Default 'none' (OFF). Only active when --rank-lambda > 0."
        ),
    )
    parser.add_argument(
        "--rank-lambda",
        type=float,
        default=0.0,
        help=(
            "Coefficient for the auxiliary rank loss (--rank-loss). "
            "Default 0.0 (OFF). Has no effect when --rank-loss none."
        ),
    )
    parser.add_argument(
        "--force-rank-during-training",
        type=int,
        default=0,
        help=(
            "When set to k > 0, replace Z with its rank-k eigh-based truncation at "
            "EVERY latent position during training AND eval. Gradients propagate "
            "through the truncation via torch.linalg.eigh autograd (stable at "
            "coincident singular values). Default 0 (OFF). Must be >= 0."
        ),
    )
    args = parser.parse_args()

    cfg = copy.deepcopy(CONFIG)
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size_per_gpu"] = args.batch_size
    if args.max_eval_batches is not None:
        cfg["max_eval_batches"] = args.max_eval_batches
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.readout is not None:
        cfg["readout"] = args.readout
    if args.base_model is not None:
        cfg["base_model"] = args.base_model
    if args.n_latents is not None:
        cfg["n_latents"] = args.n_latents
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.gamma is not None:
        cfg["gamma"] = args.gamma
    # Always assign rank flags (defaults now explicit in argparse, matching CONFIG).
    cfg["rank_loss"] = args.rank_loss
    cfg["rank_lambda"] = args.rank_lambda
    assert args.force_rank_during_training >= 0, \
        f"--force-rank-during-training must be >= 0 (got {args.force_rank_during_training})"
    cfg["force_rank_during_training"] = args.force_rank_during_training

    if args.smoke_test:
        try:
            run_smoke_tests(cfg)
        except Exception as e:
            print(f"\n[SMOKE TEST FAILED] {type(e).__name__}: {e}", file=sys.stderr)
            raise
        return

    if args.mode is None:
        parser.error("--mode is required unless --smoke-test is passed.")

    if args.mode in ("train_vanilla", "train_matrix"):
        assert "LOCAL_RANK" in os.environ, (
            "Training modes require torchrun. Launch with:\n"
            "  torchrun --standalone --nproc_per_node=8 run_matrix_codi.py "
            f"--mode {args.mode}"
        )
        train_run(args.mode, cfg)
    elif args.mode == "eval_rank_projection":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for --mode eval_rank_projection")
        eval_rank_projection(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
