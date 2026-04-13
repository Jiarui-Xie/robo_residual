from __future__ import annotations

import pytest
import torch

from robo_residual.config.residual_config import JointGroupConfig, ResidualConfig


class TestResidualConfig:
    def test_overlapping_indices_error(self):
        config = ResidualConfig(
            joint_groups=[
                JointGroupConfig("a", [0, 1, 2], 0.1),
                JointGroupConfig("b", [2, 3, 4], 0.2),  # index 2 overlaps
            ]
        )
        with pytest.raises(ValueError, match="multiple groups"):
            config.build_residual_limits(5)

    def test_limits_tensor_shape(self, simple_joint_groups):
        config = ResidualConfig(joint_groups=simple_joint_groups)
        limits = config.build_residual_limits(29)
        assert limits.shape == (29,)

    def test_limits_values_correct(self, simple_joint_groups):
        config = ResidualConfig(joint_groups=simple_joint_groups)
        limits = config.build_residual_limits(29)
        # Legs (0-11) should be 0.05
        assert (limits[:12] == 0.05).all()
        # Arms (12-28) should be 0.15
        assert (limits[12:29] == 0.15).all()

    def test_uncovered_joints_use_default(self):
        config = ResidualConfig(
            joint_groups=[JointGroupConfig("partial", [0, 1], 0.05)],
            default_max_residual=0.2,
        )
        limits = config.build_residual_limits(5)
        assert limits[0] == 0.05
        assert limits[1] == 0.05
        assert limits[2] == 0.2  # uncovered → default
        assert limits[3] == 0.2
        assert limits[4] == 0.2

    def test_out_of_range_index_error(self):
        config = ResidualConfig(
            joint_groups=[JointGroupConfig("bad", [99], 0.1)]
        )
        with pytest.raises(ValueError, match="outside action space"):
            config.build_residual_limits(10)

    def test_empty_groups_all_default(self):
        config = ResidualConfig(joint_groups=[], default_max_residual=0.3)
        limits = config.build_residual_limits(5)
        assert (limits == 0.3).all()

    def test_validate_called_by_build(self):
        config = ResidualConfig(
            joint_groups=[JointGroupConfig("neg", [-1], 0.1)]
        )
        with pytest.raises(ValueError):
            config.build_residual_limits(10)
