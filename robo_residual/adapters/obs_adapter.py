"""Observation adapter: TensorDict obs_groups → flat tensors.

rsl_rl uses TensorDict with named observation groups (e.g. "policy", "critic").
This adapter converts between the TensorDict convention and the flat tensor
interface used by ResidualActorCritic.
"""

from __future__ import annotations

from typing import Any

import torch


class ObsAdapter:
    """Extracts and concatenates observation groups from a TensorDict.

    Args:
        obs_groups: Mapping from role to list of obs group names.
            Example: ``{"policy": ["policy"], "critic": ["critic"]}``
            or: ``{"policy": ["policy", "terrain"], "critic": ["critic"]}``
    """

    def __init__(self, obs_groups: dict[str, list[str]]) -> None:
        self.obs_groups = obs_groups

    def get_actor_obs(self, obs: dict[str, torch.Tensor] | Any) -> torch.Tensor:
        """Extract and concatenate actor observation groups."""
        groups = self.obs_groups.get("policy", [])
        return self._concat_groups(obs, groups)

    def get_critic_obs(self, obs: dict[str, torch.Tensor] | Any) -> torch.Tensor:
        """Extract and concatenate critic observation groups."""
        groups = self.obs_groups.get("critic", [])
        return self._concat_groups(obs, groups)

    @staticmethod
    def _concat_groups(obs: dict[str, torch.Tensor] | Any, groups: list[str]) -> torch.Tensor:
        tensors = [obs[g] for g in groups]
        return torch.cat(tensors, dim=-1)
