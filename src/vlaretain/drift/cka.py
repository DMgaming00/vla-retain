"""Linear centered kernel alignment (Kornblith et al., 2019)."""

from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between activation matrices of shape (n_examples, n_features).

    Uses the feature-space form, ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), after centering each column.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Expected (n, d) matrices with equal n, got {tuple(x.shape)} and {tuple(y.shape)}")
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.norm(y.T @ x, ord="fro") ** 2
    denom = torch.linalg.norm(x.T @ x, ord="fro") * torch.linalg.norm(y.T @ y, ord="fro")
    if denom == 0:
        return 0.0
    return float(cross / denom)


def cka_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    return 1.0 - linear_cka(x, y)
