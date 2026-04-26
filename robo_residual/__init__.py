from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig
from robo_residual.core.onnx_base import OnnxBasePolicy
from robo_residual.core.residual_actor_critic import ResidualActorCritic
from robo_residual.core.residual_nets import GRUResidual, LSTMResidual, MLPResidual
from robo_residual.core.fuse_onnx import fuse_residual_to_onnx
from robo_residual.utils.normalizer import EmpiricalNormalization
from robo_residual.adapters.obs_adapter import ObsAdapter
from robo_residual.adapters.rsl_rl_wrapper import RslRlResidualActorCritic
from robo_residual.inference.residual_router import ResidualRouter, ResidualSlotConfig, SwitchMode
from robo_residual.robots import (
    g1_conservative,
    g1_energy_efficient,
    h1_conservative,
    h1_energy_efficient,
    go2_conservative,
    go2_energy_efficient,
)

__all__ = [
    "JointGroupConfig",
    "ResidualConfig",
    "OnnxBasePolicy",
    "ResidualActorCritic",
    "MLPResidual",
    "LSTMResidual",
    "GRUResidual",
    "fuse_residual_to_onnx",
    "EmpiricalNormalization",
    "ObsAdapter",
    "RslRlResidualActorCritic",
    "ResidualRouter",
    "ResidualSlotConfig",
    "SwitchMode",
    # pre-built robot configs
    "g1_conservative",
    "g1_energy_efficient",
    "h1_conservative",
    "h1_energy_efficient",
    "go2_conservative",
    "go2_energy_efficient",
]
