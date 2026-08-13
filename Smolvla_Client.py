"""
SmolVLA closed-loop evaluation client for Isaac Lab.

Isaac Lab stays in env_isaaclab (no lerobot install). SmolVLA runs in another
env via client/smolvla_server.py over a persistent WebSocket.

Default eval target:
  REPO_ID = "hyzhang01/GCA_suction_franka_a01_id0"
  -> task Grasp-Franka-Vacuum-IK-Rel-img, task_id a01

Terminal A (Farzam / lerobot env — loads the model):
    /home/fyp/code/Farzam/files/.venv/bin/python /home/fyp/code/Farzam/files/smolvla_server.py \
      --model_path /home/fyp/code/Farzam/files/outputs/smolvla_lora_suction_a01/checkpoints/last \
      --port 6002 --device cuda

Terminal B (env_isaaclab — runs the sim):
    python client/Smolvla_Client.py --host localhost --port 6002 --task_id a01 --num_demos 1
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import logging
import math
import pathlib
import sys
import time

import cv2
import gymnasium as gym
import numpy as np
import torch
from websockets.sync.client import connect

# Isaac Lab AppLauncher must be configured before any Isaac Lab imports.
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

# How many predicted actions to execute before the next infer request.
# 0 = full horizon (all ~50 actions the server returns).
ACTIONS_PER_QUERY = 0

parser = argparse.ArgumentParser(description="SmolVLA closed-loop control in Isaac Lab (WebSocket client).")
parser.add_argument("--task", type=str, default="Grasp-Franka-Vacuum-IK-Rel-img", help="Isaac Lab task name.")
parser.add_argument(
    "--task_id",
    type=str,
    default="a01",
    help="Task ID in task config (a01 = hyzhang01/GCA_suction_franka_a01_id0).",
)
parser.add_argument("--num_objects", type=int, default=4, help="Number of objects in scene.")
parser.add_argument("--num_demos", type=int, default=1, help="Number of evaluation episodes.")
parser.add_argument(
    "--dataset_repo_id",
    type=str,
    default="hyzhang01/GCA_suction_franka_a01_id0",
    help="LeRobot dataset repo id this policy was trained on (logged in results only).",
)
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode.")
parser.add_argument(
    "--actions_per_query",
    type=int,
    default=ACTIONS_PER_QUERY,
    help="How many chunk actions to execute before the next infer. 0 = use the full horizon (all ~50).",
)
parser.add_argument(
    "--rot_scale",
    type=float,
    default=0.5,
    help="Multiply predicted roll/pitch/yaw by this before IK. <1 reduces twist overshoot.",
)
parser.add_argument(
    "--instruction",
    type=str,
    default="",
    help="Language prompt sent to SmolVLA. Default: task config instruction for --task_id.",
)
parser.add_argument("--host", type=str, default="localhost", help="SmolVLA server host.")
parser.add_argument("--port", type=int, default=6002, help="SmolVLA server port.")
parser.add_argument("--policy", type=str, default="smolvla", help="Policy label used in output filenames.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--gripper_id", type=int, default=0, help="Gripper ID used in output filename only.")
parser.add_argument("--image_size", type=int, default=224, help="Resize camera images to this square size.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
if "Suction" in args_cli.task or "Vacuum" in args_cli.task:
    args_cli.device = "cpu"
else:
    args_cli.device = "cuda"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

_repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import isaaclab_mimic.envs  # noqa: F401
import grasp_env  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from local_utils.load_utils import TaskBuilder


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle. Adapted from robosuite."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize_uint8_image(image: np.ndarray, size: int) -> np.ndarray:
    """Match dataset cameras: HWC RGB uint8, square resize, no per-frame min-max."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        max_val = float(image.max()) if image.size else 0.0
        if max_val <= 1.0:
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(image)


def _pack_image(image: np.ndarray) -> dict:
    image = np.asarray(image, dtype=np.uint8)
    return {
        "shape": list(image.shape),
        "dtype": "uint8",
        "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
    }


class SmolVLAWebsocketPolicy:
    """Persistent WebSocket client for client/smolvla_server.py — no lerobot dependency."""

    def __init__(self, host: str, port: int, timeout_s: float = 120.0):
        self.url = f"ws://{host}:{port}"
        self.timeout_s = timeout_s
        self._ws = None
        self.metadata: dict = {}
        print(f"[SmolVLA WS Client] Connecting to {self.url}")
        self._connect()

    def _connect(self, retries: int = 30, sleep_s: float = 2.0) -> None:
        last_err = None
        for i in range(retries):
            try:
                self._ws = connect(self.url, open_timeout=5, max_size=64 * 1024 * 1024)
                raw = self._ws.recv(timeout=self.timeout_s)
                self.metadata = json.loads(raw)
                print(f"[SmolVLA WS Client] Server ready: {self.metadata}")
                return
            except Exception as e:
                last_err = str(e)
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
                print(f"[SmolVLA WS Client] Waiting for server ({i + 1}/{retries})... {last_err}")
                time.sleep(sleep_s)
        raise RuntimeError(f"SmolVLA server not reachable at {self.url}: {last_err}")

    def _request(self, payload: dict) -> dict:
        if self._ws is None:
            self._connect()
        assert self._ws is not None
        try:
            self._ws.send(json.dumps(payload))
            raw = self._ws.recv(timeout=self.timeout_s)
        except Exception:
            print("[SmolVLA WS Client] Connection dropped; reconnecting...")
            self._connect()
            assert self._ws is not None
            self._ws.send(json.dumps(payload))
            raw = self._ws.recv(timeout=self.timeout_s)
        msg = json.loads(raw)
        if msg.get("type") == "error":
            raise RuntimeError(f"Server error: {msg.get('error')}")
        return msg

    def reset(self) -> None:
        try:
            self._request({"type": "reset"})
        except Exception as e:
            print(f"[SmolVLA WS Client] reset warning: {e}")

    def infer(self, obs: dict) -> dict:
        payload = {
            "type": "infer",
            "image": _pack_image(obs["image"]),
            "wrist_image": _pack_image(obs["wrist_image"]),
            "state": np.asarray(obs["state"], dtype=np.float32).tolist(),
            "task": obs["task"],
        }
        msg = self._request(payload)
        if "actions" not in msg:
            raise RuntimeError(f"Unexpected server response: {msg}")
        actions = [np.asarray(a, dtype=np.float32) for a in msg["actions"]]
        return {"actions": actions, "server_timing": msg.get("server_timing", {})}

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def _prepare_smolvla_input(obs: dict, instruction: str, image_size: int = 224) -> dict:
    """Pack Isaac Lab observations to the same 8-D state + 224 RGB layout as the GCA dataset."""
    front_rgb = _resize_uint8_image(obs["policy"]["table_cam"][0].cpu().numpy(), image_size)
    wrist_rgb = _resize_uint8_image(obs["policy"]["wrist_cam"][0].cpu().numpy(), image_size)

    ee_pos = obs["policy"]["eef_pos"][0].cpu().numpy().astype(np.float32)
    ee_quat = obs["policy"]["eef_quat"][0].cpu().numpy()  # (w, x, y, z)
    gripper = obs["policy"]["gripper_pos"][0].cpu().numpy().astype(np.float32)

    ee_quat_xyzw = ee_quat[[1, 2, 3, 0]]
    eef_axisangle = _quat2axisangle(ee_quat_xyzw).astype(np.float32)

    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if gripper.shape[0] >= 2:
        gripper = gripper[:2]

    state = np.concatenate([ee_pos, eef_axisangle, gripper], axis=0).astype(np.float32)
    if state.shape[0] not in (7, 8):
        raise ValueError(f"Expected 7-dim (suction) or 8-dim (parallel) state, got shape {state.shape}")

    return {
        "image": front_rgb,  # HWC uint8
        "wrist_image": wrist_rgb,  # HWC uint8
        "state": state,  # (8,) float32
        "task": instruction,
    }


def check_success(env, min_height: float = 0.15) -> torch.Tensor:
    object_heights = (
        env.scene["objects"].data.object_link_pos_w[:, :, 2]
        - env.scene["objects"].data.default_object_state[:, :, 2]
    )
    return (object_heights > min_height).any(dim=1)


def check_dropping(env, min_height: float = -0.05, max_velocity: float = 50.0) -> torch.Tensor:
    object_heights = (
        env.scene["objects"].data.object_link_pos_w[:, :, 2]
        - env.scene["objects"].data.default_object_state[:, :, 2]
    )
    object_velocities = env.scene["objects"].data.object_link_vel_w[:, :, :]
    dropped = (object_heights < min_height).any(dim=1)
    velocity_magnitude = torch.norm(object_velocities, dim=2)
    too_fast = (velocity_magnitude > max_velocity).any(dim=1)
    if too_fast[0]:
        print(f"Object is moving too fast: {velocity_magnitude[0]}")
    return dropped | too_fast


def check_episode_done(env):
    success = check_success(env)
    failure = check_dropping(env)
    if failure[0]:
        success[0] = False
    return success[0] or failure[0], success[0]


def _euler_xyz_to_axangle(roll: float, pitch: float, yaw: float) -> tuple[np.ndarray, float]:
    """Convert XYZ Euler angles to axis-angle (same convention as transforms3d.euler.euler2axangle)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    if np.pi - angle < 1e-8:
        # Near 180 deg — use diagonal for axis
        xx = (R[0, 0] + 1.0) * 0.5
        yy = (R[1, 1] + 1.0) * 0.5
        zz = (R[2, 2] + 1.0) * 0.5
        axis = np.sqrt(np.maximum([xx, yy, zz], 0.0))
        if axis[0] < 1e-6:
            axis[1] = np.copysign(axis[1], R[0, 1])
            axis[2] = np.copysign(axis[2], R[0, 2])
        elif axis[1] < 1e-6:
            axis[2] = np.copysign(axis[2], R[1, 2])
        n = np.linalg.norm(axis)
        return axis / max(n, 1e-8), angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    axis = axis / (2.0 * np.sin(angle))
    return axis, angle


# Dataset abs-max on action dims 3-5 (roll, pitch, yaw).
_ROT_CLIP = np.array([0.43, 0.50, 0.50], dtype=np.float64)


def action_to_env_tensor(action7: np.ndarray, device: torch.device, rot_scale: float = 0.5) -> torch.Tensor:
    """Convert [dx, dy, dz, roll, pitch, yaw, gripper] to Isaac Lab delta pose action."""
    action7 = np.asarray(action7, dtype=np.float64).reshape(-1)
    if action7.shape[0] != 7:
        raise ValueError(f"Expected 7-dim action, got shape {action7.shape}")
    rpy = np.clip(action7[3:6] * float(rot_scale), -_ROT_CLIP, _ROT_CLIP)
    ax, ang = _euler_xyz_to_axangle(*rpy)
    delta_pose = np.concatenate([action7[0:3], ax * ang, -action7[6:7]])
    return torch.from_numpy(delta_pose).float().unsqueeze(0).to(device)


def run_episode(env, policy: SmolVLAWebsocketPolicy, max_steps: int = 500, actions_per_query: int = 10) -> dict:
    episode_start_time = time.time()
    total_steps = 0
    query_count = 0
    action_plan: collections.deque = collections.deque()

    policy.reset()
    obs, _ = env.reset()

    wait_action = torch.zeros((1, 7), device=env.device)
    for _ in range(5):
        obs, _, _, _, _ = env.step(wait_action)
        total_steps += 1

    print(f"[INFO]: Starting closed-loop control (max_steps={max_steps})")
    print(f"[INFO]: SmolVLA instruction: {env.instruction!r}")

    while total_steps < max_steps:
        episode_done, success = check_episode_done(env)
        if episode_done:
            print(f"Episode finished at step {total_steps}. Success: {success}")
            break

        try:
            if not action_plan:
                obs_fn = _prepare_smolvla_input(obs, env.instruction, image_size=args_cli.image_size)
                infer_t0 = time.time()
                action_out = policy.infer(obs_fn)
                infer_ms = 1000.0 * (time.time() - infer_t0)
                action_chunk = action_out["actions"]
                n_exec = len(action_chunk) if actions_per_query <= 0 else min(actions_per_query, len(action_chunk))
                if n_exec < 1:
                    raise RuntimeError(f"Policy returned no actions (got {len(action_chunk)}).")
                action_plan.extend(action_chunk[:n_exec])
                query_count += 1
                print(
                    f"[INFO] Query {query_count}: got {len(action_chunk)} actions, "
                    f"executing all {n_exec} before next request ({infer_ms:.1f} ms)"
                )

            action = action_plan.popleft()
            action_tensor = action_to_env_tensor(action, env.device, rot_scale=args_cli.rot_scale)
            obs, _, _, _, _ = env.step(action_tensor)
            total_steps += 1
        except Exception as e:
            print(f"Error during query {query_count}: {e}")
            import traceback

            traceback.print_exc()
            break

    final_success = bool(check_success(env)[0].item())
    return {
        "success": final_success,
        "total_steps": total_steps,
        "queries": query_count,
        "execution_time_seconds": time.time() - episode_start_time,
    }


def main() -> None:
    policy = SmolVLAWebsocketPolicy(host=args_cli.host, port=args_cli.port)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_builder = TaskBuilder(task_id=args_cli.task_id)
    env_cfg.scene.objects = task_builder.create_scene_objects(num_objs=args_cli.num_objects)
    env_cfg.terminations.time_out = None
    if args_cli.task_id.startswith("a"):
        env_cfg.scene.table.spawn.scale = [0.7, 1.0, 1.0]

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.seed(args_cli.seed)
    env.instruction = args_cli.instruction.strip() or task_builder.task_instruction
    if env.instruction != "pick up craker box from stacked boxes":
        print(
            f"[WARN]: instruction {env.instruction!r} does not match the b01 training string "
            "'pick up craker box from stacked boxes'"
        )

    results = []
    success_count = 0
    print(f"[INFO]: Running {args_cli.num_demos} closed-loop demonstrations")
    print(f"[INFO]: Instruction: {env.instruction}")
    print(f"[INFO]: Server: {args_cli.host}:{args_cli.port}")
    print(f"[INFO]: Max steps per episode: {args_cli.max_steps}")
    print(f"[INFO]: Actions per query: {args_cli.actions_per_query}")
    print(f"[INFO]: Rotation scale: {args_cli.rot_scale} (clip={_ROT_CLIP.tolist()})")

    for demo_idx in range(args_cli.num_demos):
        print(f"\n{'=' * 50}")
        print(f"Demo {demo_idx + 1}/{args_cli.num_demos}")
        print(f"{'=' * 50}")
        try:
            result = run_episode(
                env=env,
                policy=policy,
                max_steps=args_cli.max_steps,
                actions_per_query=args_cli.actions_per_query,
            )
            result["demo_idx"] = demo_idx
            if result["success"]:
                success_count += 1
                print(
                    f"✓ Demo {demo_idx + 1} SUCCESS in {result['total_steps']} steps "
                    f"with {result['queries']} queries ({result['execution_time_seconds']:.2f}s)"
                )
            else:
                print(
                    f"✗ Demo {demo_idx + 1} FAILED after {result['total_steps']} steps "
                    f"with {result['queries']} queries ({result['execution_time_seconds']:.2f}s)"
                )
            results.append(result)
        except Exception as e:
            print(f"Error in demo {demo_idx + 1}: {e}")
            results.append(
                {
                    "demo_idx": demo_idx,
                    "success": False,
                    "error": str(e),
                    "total_steps": 0,
                    "queries": 0,
                    "execution_time_seconds": 0,
                }
            )

    print(f"\n{'=' * 50}")
    print("FINAL RESULTS")
    print(f"{'=' * 50}")
    print(f"Success rate: {success_count}/{args_cli.num_demos} ({100 * success_count / args_cli.num_demos:.1f}%)")

    avg_time_all = 0.0
    total_time_all = 0.0
    avg_time_success = 0.0
    successful_results = [r for r in results if r.get("success")]
    if results:
        all_execution_times = [r.get("execution_time_seconds", 0) for r in results]
        avg_time_all = float(np.mean(all_execution_times))
        total_time_all = float(np.sum(all_execution_times))
        if successful_results:
            avg_steps = float(np.mean([r["total_steps"] for r in successful_results]))
            avg_queries = float(np.mean([r["queries"] for r in successful_results]))
            avg_time_success = float(np.mean([r.get("execution_time_seconds", 0) for r in successful_results]))
            print(f"Average steps (successful): {avg_steps:.1f}")
            print(f"Average queries (successful): {avg_queries:.1f}")
            print(f"Average execution time (successful): {avg_time_success:.2f}s")
        print(f"\nAverage execution time (all episodes): {avg_time_all:.2f}s")
        print(f"Total execution time (all episodes): {total_time_all:.2f}s")

    output_dir = pathlib.Path("json_result/adaptation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args_cli.policy}_{args_cli.gripper_id}_{args_cli.task_id}_d{args_cli.num_demos}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "dataset_repo_id": args_cli.dataset_repo_id,
                "task_id": args_cli.task_id,
                "task": args_cli.task,
                "server": f"{args_cli.host}:{args_cli.port}",
                "num_objects": args_cli.num_objects,
                "num_demos": args_cli.num_demos,
                "max_steps": args_cli.max_steps,
                "actions_per_query": args_cli.actions_per_query,
                "success_rate": success_count / args_cli.num_demos,
                "avg_execution_time_all": avg_time_all,
                "total_execution_time": total_time_all,
                "avg_execution_time_successful": avg_time_success if successful_results else 0,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {output_file}")
    env.close()
    policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    simulation_app.close()
