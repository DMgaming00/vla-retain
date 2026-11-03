"""Policy divergence from S1 on probe states. All metrics share noise, so a policy equal to S1 scores exactly 0."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from vlaretain.drift.cka import cka_distance
from vlaretain.models.load import lm_hidden_states, pool_hidden, predict_velocity, sample_action_chunk


@dataclass(frozen=True)
class DivergenceConfig:
    n_action_samples: int = 16
    n_velocity_draws: int = 4
    seed: int = 0


def _generator(seed: int, *parts: int) -> torch.Generator:
    value = seed
    for part in parts:
        value = (value * 1_000_003 + part) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(value)


def _padded_shape(policy, n: int) -> tuple[int, int, int]:
    return (n, policy.config.chunk_size, policy.config.max_action_dim)


def _batch_size(batch: dict) -> int:
    return batch["observation.state"].shape[0]


def _device(policy) -> torch.device:
    return next(policy.parameters()).device


@torch.no_grad()
def shared_noise_action_distance(policy, reference, batches: Iterable[dict], cfg: DivergenceConfig) -> float:
    """Mean over frames and chunk steps of ||a_theta(o, eps) - a_S1(o, eps)||_2, in normalized action space."""
    total, count = 0.0, 0
    device = _device(policy)
    for index, batch in enumerate(batches):
        n = _batch_size(batch)
        noise = torch.randn(_padded_shape(policy, n), generator=_generator(cfg.seed, 1, index)).to(device)
        a = sample_action_chunk(policy, batch, noise)
        b = sample_action_chunk(reference, batch, noise)
        total += torch.linalg.vector_norm(a - b, dim=-1).mean(dim=1).sum().item()
        count += n
    return total / max(count, 1)


@torch.no_grad()
def velocity_distance(policy, reference, batches: Iterable[dict], cfg: DivergenceConfig) -> float:
    """E ||v_theta - v_S1||^2 at random t and shared x_t, over valid action dims and non-padded chunk steps."""
    action_dim = policy.config.output_features["action"].shape[0]
    total, count = 0.0, 0
    device = _device(policy)
    for index, batch in enumerate(batches):
        n = _batch_size(batch)
        actions = policy.prepare_action(batch)
        valid = ~batch["action_is_pad"]
        for draw in range(cfg.n_velocity_draws):
            g = _generator(cfg.seed, 2, index, draw)
            noise = torch.randn(_padded_shape(policy, n), generator=g).to(device)
            # Uniform over the training support of t.
            time = (torch.rand(n, generator=g) * 0.999 + 0.001).to(device)
            x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
            diff = predict_velocity(policy, batch, x_t, time) - predict_velocity(reference, batch, x_t, time)
            per_step = (diff[:, :, :action_dim] ** 2).sum(dim=-1)
            total += (per_step * valid).sum().item()
            count += valid.sum().item()
    return total / max(count, 1)


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Biased (V-statistic) squared MMD with an RBF kernel at the median-heuristic bandwidth. Always >= 0.

    The bandwidth is the median pairwise squared distance of the pooled samples. Identical samples give exactly 0.
    """
    pooled = torch.cat([x, y]).to(torch.float64)
    x, y = pooled[: len(x)], pooled[len(x) :]
    squared = torch.cdist(pooled, pooled) ** 2
    off_diagonal = squared[~torch.eye(len(pooled), dtype=torch.bool, device=pooled.device)]
    bandwidth = off_diagonal.median().clamp_min(1e-12)

    def kernel(a, b):
        return torch.exp(-(torch.cdist(a, b) ** 2) / bandwidth)

    value = kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()
    return value.clamp_min(0.0)


@torch.no_grad()
def action_mmd(policy, reference, batches: Iterable[dict], cfg: DivergenceConfig) -> float:
    """Mean over states of MMD^2 between 16 action-chunk samples from each policy.

    Both policies get the same 16 noise draws, so a policy equal to S1 scores 0 rather than sampling noise.
    """
    total, count = 0.0, 0
    k = cfg.n_action_samples
    device = _device(policy)
    for index, batch in enumerate(batches):
        n = _batch_size(batch)
        expanded = {
            key: value.repeat_interleave(k, dim=0) if torch.is_tensor(value) else value for key, value in batch.items()
        }
        noise = torch.randn(_padded_shape(policy, n * k), generator=_generator(cfg.seed, 3, index)).to(device)
        a = sample_action_chunk(policy, expanded, noise).reshape(n, k, -1)
        b = sample_action_chunk(reference, expanded, noise).reshape(n, k, -1)
        for i in range(n):
            total += rbf_mmd2(a[i], b[i]).item()
        count += n
    return total / max(count, 1)


@torch.no_grad()
def cka_per_layer(policy, reference, batches: Iterable[dict]) -> dict[int, float]:
    """1 - linear CKA per LM layer, on hidden states pooled over non-padding tokens."""
    ours: dict[int, list[torch.Tensor]] = {}
    theirs: dict[int, list[torch.Tensor]] = {}
    for batch in batches:
        hidden_a, pad = lm_hidden_states(policy, batch)
        hidden_b, _ = lm_hidden_states(reference, batch)
        for layer in hidden_a:
            ours.setdefault(layer, []).append(pool_hidden(hidden_a[layer], pad).cpu())
            theirs.setdefault(layer, []).append(pool_hidden(hidden_b[layer], pad).cpu())
    return {layer: cka_distance(torch.cat(ours[layer]), torch.cat(theirs[layer])) for layer in sorted(ours)}


def compute_probe_metrics(policy, reference, batches: list[dict], cfg: DivergenceConfig) -> dict:
    """Every state-dependent metric, for one probe split."""
    return {
        "action_chunk_distance": shared_noise_action_distance(policy, reference, batches, cfg),
        "velocity_distance": velocity_distance(policy, reference, batches, cfg),
        "action_mmd": action_mmd(policy, reference, batches, cfg),
        "cka_distance": cka_per_layer(policy, reference, batches),
    }
