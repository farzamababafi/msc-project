"""
Stage 4 (decisive test): Replay a real TRAINING frame through the live
smolvla_server.py, and compare the predicted action chunk against the
dataset's actual recorded ground-truth actions for that same episode.

This bypasses the sim entirely. It answers one question directly:

    Does the model predict sensibly on data it was trained on?

    YES -> the model itself learned something real. The bug is in the
           sim<->server bridge: image format/domain, state convention,
           units, or task string. Look there next, not at epochs/rank.

    NO  -> the model itself didn't learn the mapping properly, even on
           its own training distribution. That's a training-side bug
           (bad normalization stats, broken conditioning, wrong feature
           mapping) -- more epochs will just memorize it more
           confidently, won't fix it.

PREREQUISITE: smolvla_server.py must already be running (bash serve.sh),
reachable at --host/--port below.

Usage:
    python 04_replay_sanity_check.py \
        --host 127.0.0.1 --port 6002 \
        --episode 0 --frame_offset 0 --compare_steps 10
"""

from __future__ import annotations

import argparse
import json

import msgpack
import numpy as np
import torch
from websockets.sync.client import connect

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "hyzhang01/GCA_parallel_adaptation_b01"


def _pack_default(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot msgpack-encode type {type(obj)}")


def _unpack_hook(obj):
    if isinstance(obj, dict) and obj.get("__ndarray__") is True:
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def chw_float_to_hwc_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Matches what the server expects on the wire: HWC uint8."""
    arr = (img_tensor.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).contiguous().numpy()
    return arr


def get_task_string(dataset: LeRobotDataset, sample: dict) -> str:
    if "task" in sample and isinstance(sample["task"], str):
        return sample["task"]
    task_index = int(sample.get("task_index", 0))
    tasks = dataset.meta.tasks
    if isinstance(tasks, dict):
        # {task_index: task_string} or {task_string: task_index} depending on version
        for k, v in tasks.items():
            if k == task_index or v == task_index:
                return v if isinstance(v, str) else k
    print(f"WARNING: could not resolve task string for task_index={task_index}, sending empty string.")
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6002)
    parser.add_argument("--episode", type=int, default=0, help="Which training episode to replay from")
    parser.add_argument("--frame_offset", type=int, default=0, help="Frame within that episode to start at")
    parser.add_argument("--compare_steps", type=int, default=10, help="How many ground-truth steps to compare")
    args = parser.parse_args()

    print(f"Loading {REPO_ID}, episode {args.episode} ...")
    dataset = LeRobotDataset(REPO_ID, episodes=[args.episode])

    ep_start = dataset.episode_data_index["from"][0].item()
    ep_end = dataset.episode_data_index["to"][0].item()
    idx = ep_start + args.frame_offset

    if idx + args.compare_steps >= ep_end:
        raise ValueError(
            f"episode {args.episode} too short for frame_offset={args.frame_offset} "
            f"+ compare_steps={args.compare_steps}. Episode has {ep_end - ep_start} frames."
        )

    sample = dataset[idx]
    task_str = get_task_string(dataset, sample)
    print(f"Task string being sent: {task_str!r}")

    image = chw_float_to_hwc_uint8(sample["observation.images.image"])
    wrist_image = chw_float_to_hwc_uint8(sample["observation.images.wrist_image"])
    state = sample["observation.state"].numpy().astype(np.float32)

    print(f"State sent to server: {state}")

    # Ground truth: the next `compare_steps` recorded actions from this same episode
    gt_actions = []
    for k in range(args.compare_steps):
        gt_actions.append(dataset[idx + k]["action"].numpy())
    gt_actions = np.stack(gt_actions)  # (compare_steps, action_dim)

    print(f"Connecting to ws://{args.host}:{args.port} ...")
    with connect(f"ws://{args.host}:{args.port}") as ws:
        metadata_raw = ws.recv()
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else msgpack.unpackb(
            metadata_raw, object_hook=_unpack_hook, raw=False
        )
        print(f"Server metadata: {metadata}")

        ws.send(msgpack.packb({"type": "reset"}, default=_pack_default, use_bin_type=True))
        ws.recv()  # ack

        payload = {
            "type": "infer",
            "image": image,
            "wrist_image": wrist_image,
            "state": state,
            "task": task_str,
        }
        ws.send(msgpack.packb(payload, default=_pack_default, use_bin_type=True))
        response_raw = ws.recv()
        response = msgpack.unpackb(response_raw, object_hook=_unpack_hook, raw=False)

    if response.get("type") == "error":
        print(f"SERVER ERROR: {response['error']}")
        return

    pred_actions = np.stack(response["actions"][: args.compare_steps])

    print("\n=== Comparison: predicted vs. ground-truth actions ===")
    print(f"{'step':>4}  {'pred':>40}  {'ground truth':>40}")
    for k in range(args.compare_steps):
        print(f"{k:>4}  {np.array2string(pred_actions[k], precision=3):>40}  "
              f"{np.array2string(gt_actions[k], precision=3):>40}")

    abs_error = np.abs(pred_actions - gt_actions)
    print(f"\nMean abs error across {args.compare_steps} steps: {abs_error.mean():.4f}")
    print(f"Per-dimension mean abs error: {abs_error.mean(axis=0)}")

    print(
        "\nInterpretation:\n"
        "  - Small, consistent error across dims -> model predicts sensibly on its own\n"
        "    training data. The bug is in the sim<->server bridge (image domain, state\n"
        "    convention, task string) -- NOT in training epochs or LoRA rank.\n"
        "  - Large error, or error concentrated in one specific dimension (e.g. one\n"
        "    axis or the gripper channel) -> that dimension is a good place to check\n"
        "    for a sign flip, unit mismatch, or normalization bug in TRAINING itself.\n"
        "  - Large error across all dimensions -> training-side bug (stats, "
        "conditioning, or feature mapping), not a sim-serving bridge problem."
    )


if __name__ == "__main__":
    main()
