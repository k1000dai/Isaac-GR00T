"""Policy server for GR00T."""

import argparse
import contextlib
import json
import os
import pathlib
import socket
import time
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.policy import BasePolicy
import numpy as np
import pyarrow as pa


def _load_policy(model_path, embodiment_tag):
    return Gr00tPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda:0",
        strict=True,
    )


def _debug_image(name: str, img: np.ndarray) -> None:
    print(
        f"  [DBG] image '{name}': shape={img.shape} dtype={img.dtype} "
        f"min={img.min()} max={img.max()} mean={img.mean():.1f}"
    )
    if img.dtype != np.uint8:
        print(f"  [DBG] WARNING: expected uint8 for '{name}', got {img.dtype}")
    if img.shape[-1] != 3:
        print(f"  [DBG] WARNING: expected 3-channel image for '{name}', got {img.shape[-1]} channels")


def _parse_observations(observations, metadata, debug: bool = False):
    last_observation = observations[-1]
    pos_shape = len(last_observation["position"])

    def build_camera_observation(name):
        return (
            last_observation[name]
            .values.to_numpy(zero_copy_only=False)
            .reshape(
                metadata[f"{name}.height"],
                metadata[f"{name}.width"],
                3,
            )
        )

    field_names = [field.name for field in observations.type.fields]
    if debug:
        print(f"[DBG] observation fields: {field_names}")
        print(f"[DBG] metadata: {metadata}")

    cameras = {}
    if "camera_wrist_right" in field_names:
        cameras["wrist_right"] = build_camera_observation("camera_wrist_right")
    if "camera_wrist_left" in field_names:
        cameras["wrist_left"] = build_camera_observation("camera_wrist_left")
    cameras["ceiling"] = build_camera_observation("camera_ceiling")
    cameras["head_left"] = build_camera_observation("camera_head_left")
    cameras["head_right"] = build_camera_observation("camera_head_right")

    timestamp = metadata["timestamp"]
    qpos = (
        last_observation["position"]
        .values.to_numpy(zero_copy_only=False)
        .reshape(pos_shape)
    )

    if debug:
        print(f"[DBG] timestamp={timestamp}")
        print(f"[DBG] qpos shape={qpos.shape} values={qpos}")
        for cam_name, img in cameras.items():
            _debug_image(cam_name, img)

    return timestamp, qpos, cameras


def _infer(
    adapter,
    timestamp,
    qpos,
    frames,
    language_instruction: str,
    debug: bool = False,
):
    inference_hz = 30

    last_chunk = adapter.get_action_chunk(frames, qpos, language_instruction)

    if debug:
        print(f"[DBG] action chunk keys: {list(last_chunk.keys())}")
        for k, v in last_chunk.items():
            arr = np.asarray(v)
            print(f"  [DBG] chunk['{k}']: shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")

    positions = adapter.action_chunk_to_qpos_sequence(last_chunk, qpos)

    if debug:
        print(f"[DBG] positions sequence: shape={positions.shape}")
        print(f"[DBG] positions[0]={positions[0]}")
        print(f"[DBG] positions[-1]={positions[-1]}")

    action = {
        "interval": int(1e9 / inference_hz),
        "cutoff_hz": 5,
        "positions": positions.tolist(),
    }
    return action


def _process_request(io, adapter, infer_hz=10, language_instruction: str = "", debug: bool = False):
    print(f"Processing request with infer_hz: {infer_hz}", flush=True)

    INTERVAL_NS = int(1e9 / infer_hz)
    RESET_INTERVAL_NS = 1e9
    last_infer_time_ns = time.monotonic_ns()
    SUMMARY_INTERVAL_NS = int(5e9)  # print summary every 5 s
    last_summary_ns = time.monotonic_ns()

    n_received = 0
    n_throttled = 0
    n_inferred = 0

    print("Waiting for first request...", flush=True)
    for request_json in io:
        if n_received == 0:
            print(f"First raw line received ({len(request_json)} bytes): {request_json[:120]!r}", flush=True)

        try:
            request = json.loads(request_json)
        except json.JSONDecodeError:
            print(f"Received invalid JSON: {request_json[:200]!r}", flush=True)
            continue

        n_received += 1
        current_time_ns = time.monotonic_ns()
        elapsed = current_time_ns - last_infer_time_ns

        # Periodic summary (always printed, not gated on --debug)
        if current_time_ns - last_summary_ns >= SUMMARY_INTERVAL_NS:
            dt = (current_time_ns - last_summary_ns) / 1e9
            print(
                f"[stats] {dt:.1f}s  received={n_received}  "
                f"inferred={n_inferred}  throttled={n_throttled}"
            )
            n_received = n_throttled = n_inferred = 0
            last_summary_ns = current_time_ns

        if elapsed < INTERVAL_NS:
            n_throttled += 1
            if debug:
                print(
                    f"[DBG] throttled (elapsed={elapsed/1e6:.1f}ms < "
                    f"{INTERVAL_NS/1e6:.1f}ms)"
                )
            io.write(json.dumps({"positions": []}) + "\n")
            io.flush()
            continue

        if elapsed > RESET_INTERVAL_NS:
            print(f"[stats] resetting policy (gap={elapsed/1e9:.2f}s)")
            adapter.reset()

        last_infer_time_ns = current_time_ns
        n_inferred += 1

        metadata = request["metadata"]
        if debug:
            print(f"[DBG] data_path={request.get('data_path')}")

        with pa.OSFile(request["data_path"], "rb") as source:
            with pa.ipc.open_file(source) as reader:
                observations = reader.get_batch(0).to_struct_array()
                obs = _parse_observations(observations, metadata, debug=debug)
                actions = _infer(adapter, *obs, language_instruction=language_instruction, debug=debug)
                if debug:
                    print(f"[DBG] sending {len(actions['positions'])} position steps, cutoff_hz={actions['cutoff_hz']}")
                io.write(json.dumps(actions) + "\n")
                io.flush()

    print(f"Connection closed (received={n_received} inferred={n_inferred} throttled={n_throttled})", flush=True)


class OpenArmGR00TAdapter:
    """OpenArmGR00TAdapter.

    Adapter to convert between:
    - OpenArm robot observations → GR00T Policy input format
    - GR00T Policy action output → OpenArm motor commands

    Modality keys (must match your dataset's modality.json):
        video:   wrist_right, wrist_left, ceiling, head_left, head_right
        state:   right_arm (7), right_gripper (1), left_arm (7), left_gripper (1)
        action:  right_arm (7), right_gripper (1), left_arm (7), left_gripper (1)
        language: annotation.human.action.task_description
    """

    def __init__(
        self,
        policy: BasePolicy,
        state_dim: int = 16,
        language_key: str = "annotation.human.action.task_description",
    ):
        """OpenArmGR00TAdapter.

        Args:
        policy: Policy implementation, such as Gr00tPolicy or PolicyClient.
        state_dim: Robot state dimension (8 for single-arm, 16 for bimanual)
        language_key: The language annotation key in modality config

        """
        self.policy = policy
        self.state_dim = state_dim
        self.language_key = language_key
        self.bimanual = state_dim == 16

    def split_qpos(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """Split qpos into state keys matching modality.json.

        For bimanual (16 DoF):
            qpos[0:7]  -> right_arm
            qpos[7:8]  -> right_gripper
            qpos[8:15] -> left_arm
            qpos[15:16] -> left_gripper

        For single-arm (8 DoF):
            qpos[0:7]  -> right_arm
            qpos[7:8]  -> right_gripper
        """
        qpos = np.asarray(qpos, dtype=np.float32)

        if self.bimanual:
            if qpos.shape[0] != 16:
                raise ValueError(f"Expected 16-DoF qpos for bimanual, got {qpos.shape}")
            return {
                "right_arm": qpos[0:7],
                "right_gripper": qpos[7:8],
                "left_arm": qpos[8:15],
                "left_gripper": qpos[15:16],
            }
        else:
            if qpos.shape[0] != 8:
                raise ValueError(
                    f"Expected 8-DoF qpos for single-arm, got {qpos.shape}"
                )
            return {
                "right_arm": qpos[0:7],
                "right_gripper": qpos[7:8],
            }

    def merge_action_to_qpos(
        self,
        action_chunk: dict[str, np.ndarray],
        t: int,
        qpos_current: np.ndarray,
    ) -> np.ndarray:
        """Merge action chunk at timestep t into a full qpos command.

        Missing keys fall back to current state.

        Args:
            action_chunk: Dict of action arrays with shape (B=1, T, D)
            t: Timestep index into the action horizon
            qpos_current: Current robot joint positions

        """
        qpos_current = np.asarray(qpos_current, dtype=np.float32)

        def get_or_current(key: str, cur: np.ndarray) -> np.ndarray:
            if key not in action_chunk:
                return cur
            return np.asarray(action_chunk[key][0, t], dtype=np.float32)

        if self.bimanual:
            cur_right = qpos_current[:8].copy()
            cur_left = qpos_current[8:16].copy()

            right_arm = get_or_current("right_arm", cur_right[:7])
            right_gripper = get_or_current("right_gripper", cur_right[7:8])
            left_arm = get_or_current("left_arm", cur_left[:7])
            left_gripper = get_or_current("left_gripper", cur_left[7:8])

            return np.concatenate([right_arm, right_gripper, left_arm, left_gripper])
        else:
            cur = qpos_current.copy()
            right_arm = get_or_current("right_arm", cur[:7])
            right_gripper = get_or_current("right_gripper", cur[7:8])
            return np.concatenate([right_arm, right_gripper])

    def obs_to_policy_input(
        self,
        frames: dict[str, np.ndarray],
        qpos: np.ndarray,
        language_instruction: str,
    ) -> dict[str, Any]:
        """Convert robot observations into GR00T Policy input format.

        Args:
            frames: Dict of camera nickname -> np.ndarray
            qpos: Current joint positions
            language_instruction: Task instruction string

        Returns:
            Dict with structure:
            {
                "video": {key: (B=1, T=1, H, W, 3) uint8},
                "state": {key: (B=1, T=1, D) float32},
                "language": {key: [[str]]}
            }

        """
        model_obs: dict[str, Any] = {}

        # (1) Video: HWC, uint8
        model_obs["video"] = frames

        # (2) State: split qpos into state keys
        model_obs["state"] = self.split_qpos(qpos)

        # (3) Language
        model_obs["language"] = {self.language_key: language_instruction}

        # (4) Add (B=1, T=1) dimensions
        model_obs = self._recursive_add_extra_dim(model_obs)
        model_obs = self._recursive_add_extra_dim(model_obs)

        return model_obs

    def get_action_chunk(
        self,
        frames: dict[str, np.ndarray],
        qpos: np.ndarray,
        language_instruction: str,
    ) -> dict[str, np.ndarray]:
        """Query policy and return action chunk."""
        model_input = self.obs_to_policy_input(frames, qpos, language_instruction)
        action_chunk, info = self.policy.get_action(model_input)
        return action_chunk

    @staticmethod
    def infer_horizon(action_chunk: dict[str, np.ndarray]) -> int:
        """Infer action horizon from action chunk shape."""
        any_key = next(iter(action_chunk.keys()))
        return int(action_chunk[any_key].shape[1])

    def action_chunk_to_qpos_sequence(
        self, action_chunk: dict[str, np.ndarray], qpos_current: np.ndarray
    ) -> np.ndarray:
        """Convert a full action chunk into a (T, state_dim) qpos sequence."""
        horizon = self.infer_horizon(action_chunk)
        qpos_seq = np.zeros((horizon, self.state_dim), dtype=np.float32)
        for t in range(horizon):
            qpos_seq[t] = self.merge_action_to_qpos(action_chunk, t, qpos_current)
        return qpos_seq

    def reset(self):
        """Reset policy."""
        self.policy.reset()

    def _recursive_add_extra_dim(self, obs: dict) -> dict:
        """Recursively add an extra dimension to arrays or scalars.

        GR00T Policy expects: obs with shape (batch=1, time=1, ...).
        Call this function twice to achieve (B=1, T=1, ...).
        """
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs[key] = val[np.newaxis, ...]
            elif isinstance(val, dict):
                obs[key] = self._recursive_add_extra_dim(val)
            else:
                obs[key] = [val]  # scalar → [scalar]
        return obs


def main():
    """Infer the next actions from observations."""
    parser = argparse.ArgumentParser(
        description="GR00T for OpenArm",
    )
    parser.add_argument("--socket-path",
        default=os.getenv("SOCKET_PATH", "/dev/shm/policy-server.socket"),
        help="Path to the UNIX socket",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=os.getenv("CHECKPOINT_FILE"),
        help="The checkpoint file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--embodiment-tag",
        default=os.getenv("EMBODIMENT_TAG", "openarm_bimanual_rel_all_cam"),
        type=str,
        help="openarm_bimanual_rel_all_cam/openarm_bimanual_rel",
    )
    parser.add_argument(
        "--local-server",
        action="store_true",
        help="Run as a local policy server."
    )
    parser.add_argument(
        "--infer-hz",
        default=10.0,
        type=float,
        help="Inference frequency in Hz"
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv("PROMPT", "Wipe the spill on the tray."),
        type=str,
        help="Language instruction passed to the policy",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info: image shapes/ranges, qpos, action chunks",
    )

    args = parser.parse_args()

    policy = _load_policy(args.checkpoint_file, args.embodiment_tag)
    adapter = OpenArmGR00TAdapter(policy=policy)

    policy.reset()

    if args.local_server:
        try:
            print(f"Listening socket: {args.socket_path}")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.bind(args.socket_path)
                sock.listen()
                while True:
                    conn, addr = sock.accept()
                    print(f"Client connected: {addr}", flush=True)
                    with conn, conn.makefile("rw") as io:
                        _process_request(io, adapter, args.infer_hz, language_instruction=args.prompt, debug=args.debug)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(args.socket_path)
    else:
        print(f"Connecting to socket: {args.socket_path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(args.socket_path)
            with sock.makefile("rw") as io:
                _process_request(io, adapter, args.infer_hz, language_instruction=args.prompt, debug=args.debug)


if __name__ == "__main__":
    main()
