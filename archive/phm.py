"""
Parameterized Hypercomplex Multiplication (PHM) layer.

Three modes:
1. 'learned' — A matrices are fully learnable (original PHM)
2. 'quaternion' — A matrices are FIXED to Hamilton quaternion basis
3. 'quaternion_init' — A matrices START as quaternion basis but can learn

Reference: "Beyond Fully-Connected Layers with Quaternions" (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math


def quaternion_basis():
    """Hamilton quaternion basis matrices {1, i, j, k} as (4, 4, 4) tensor."""
    A = torch.zeros(4, 4, 4)
    # 1 (identity)
    A[0] = torch.eye(4)
    # i: i²=-1, ij=k, ik=-j
    A[1] = torch.tensor([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,-1.],[0.,0.,1.,0.]])
    # j: j²=-1, ji=-k, jk=i
    A[2] = torch.tensor([[0.,0.,-1.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.],[0.,-1.,0.,0.]])
    # k: k²=-1, ki=j, kj=-i
    A[3] = torch.tensor([[0.,0.,0.,-1.],[0.,0.,-1.,0.],[0.,1.,0.,0.],[1.,0.,0.,0.]])
    return A


class PHMLinear(nn.Module):
    """Linear layer using Parameterized Hypercomplex Multiplication.

    W = sum_i (A_i ⊗ S_i) where A_i are (n,n) algebra matrices
    and S_i are (out/n, in/n) content matrices.

    Args:
        algebra_mode: 'learned' (default), 'quaternion' (fixed), 'quaternion_init' (warm-start)
    """

    def __init__(self, in_features: int, out_features: int, n: int = 4,
                 bias: bool = True, algebra_mode: str = 'learned'):
        super().__init__()
        assert in_features % n == 0
        assert out_features % n == 0

        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.algebra_mode = algebra_mode

        if algebra_mode == 'quaternion':
            # Fixed quaternion basis — not learnable
            assert n == 4, "Quaternion mode requires n=4"
            self.register_buffer('A', quaternion_basis())
        elif algebra_mode == 'quaternion_init':
            # Start from quaternion basis, allow learning
            assert n == 4, "Quaternion init mode requires n=4"
            self.A = nn.Parameter(quaternion_basis())
        else:
            # Fully learned (original PHM)
            self.A = nn.Parameter(torch.empty(n, n, n))

        self.S = nn.Parameter(torch.empty(n, out_features // n, in_features // n))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.algebra_mode == 'learned':
            # Initialize A as near-identity (original PHM init)
            for i in range(self.n):
                init.zeros_(self.A[i])
                self.A.data[i, i, i] = 1.0 / self.n
        # For quaternion and quaternion_init, A is already set

        # Initialize S with Kaiming
        for i in range(self.n):
            init.kaiming_uniform_(self.S[i], a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._kronecker_sum()
        out = x @ W.t()
        if self.bias is not None:
            out = out + self.bias
        return out

    def _kronecker_sum(self) -> torch.Tensor:
        W = torch.zeros(self.out_features, self.in_features,
                        device=self.A.device, dtype=self.A.dtype)
        for i in range(self.n):
            W = W + torch.kron(self.A[i], self.S[i])
        return W

    def get_algebra(self) -> torch.Tensor:
        return self.A.detach().clone()


class PHMLP(nn.Module):
    """Two-layer MLP using PHM layers."""

    def __init__(self, d_model: int, d_ff: int, n: int = 4, dropout: float = 0.1,
                 algebra_mode: str = 'learned'):
        super().__init__()
        self.fc1 = PHMLinear(d_model, d_ff, n=n, algebra_mode=algebra_mode)
        self.fc2 = PHMLinear(d_ff, d_model, n=n, algebra_mode=algebra_mode)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))
