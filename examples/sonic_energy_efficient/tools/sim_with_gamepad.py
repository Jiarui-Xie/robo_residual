"""Run gear_sonic's MuJoCo sim loop with a *gamepad injector* so that deploy
gets forward-velocity commands over DDS instead of via keyboard stdin.

Why this exists
---------------
deploy.sh (with --input-type manager or --input-type gamepad) reads the
joystick packet from `low_state.wireless_remote` (40 bytes embedded inside
every LowState_ message the sim publishes).  gear_sonic's
`UnitreeSdk2Bridge.PublishLowState` never writes those bytes, so deploy's
gamepad reads all zeros → robot stays idle.

This script:
  1. Monkey-patches `UnitreeSdk2Bridge.PublishLowState` to copy a 40-byte
     "gamepad frame" into `self.low_state.wireless_remote` right before it
     calls `Write`.  The frame is read from a shared file
     (`/tmp/wireless_remote_inject.bin`) each tick — the recorder process
     writes it.
  2. Delegates to `gear_sonic.scripts.run_sim_loop.main` so the rest of
     the sim behaviour is identical.

Run it in the .venv_sim exactly like run_sim_loop:

    source /home/lumi/GR00T-WholeBodyControl/.venv_sim/bin/activate
    python -m examples.sonic_energy_efficient.tools.sim_with_gamepad
        # (same CLI flags as gear_sonic.scripts.run_sim_loop; defaults work)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

INJECT_PATH = Path(os.environ.get("WIRELESS_REMOTE_INJECT", "/tmp/wireless_remote_inject.bin"))


def _install_gamepad_patch() -> None:
    """Replace UnitreeSdk2Bridge.PublishLowState so each tick it reads the
    current 40-byte gamepad frame from INJECT_PATH and stores it into
    `self.low_state.wireless_remote` before publishing.
    """
    from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as bridge_mod

    Bridge = bridge_mod.UnitreeSdk2Bridge
    _original_publish = Bridge.PublishLowState

    def _read_injected_bytes() -> bytes | None:
        try:
            data = INJECT_PATH.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        if len(data) != 40:
            return None
        return data

    def PublishLowState_with_gamepad(self, obs):  # type: ignore[override]
        wr = _read_injected_bytes()
        if wr is not None:
            # `wireless_remote` is a 40-element uint8 array on the IDL dataclass.
            self.low_state.wireless_remote[:] = list(wr)
        return _original_publish(self, obs)

    Bridge.PublishLowState = PublishLowState_with_gamepad
    print(f"[sim_with_gamepad] wireless_remote injection enabled from {INJECT_PATH}",
          flush=True)


def main() -> None:
    _install_gamepad_patch()

    # Defer import until AFTER the patch so the patched method is picked up.
    from gear_sonic.scripts import run_sim_loop

    import tyro

    config = tyro.cli(run_sim_loop.ArgsConfig)
    run_sim_loop.main(config)


if __name__ == "__main__":
    sys.exit(main())
