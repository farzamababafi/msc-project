"""
Stage 5: Open-loop evaluation on held-out episodes.

Compares the fine-tuned policy's predicted actions against recorded
ground-truth actions on episodes NOT used in training. This is a cheap
sanity check to run BEFORE testing on a real arm — it catches gross
failures (bad normalization, wrong camera key mapping, exploded loss)
without needing hardware.

It does NOT catch compounding drift (errors accumulating over a closed
control loop) — for that you need real or simulated rollout, not replay.

Usage:
    # latest checkpoint (symlink created by lerobot-train)
    python 03_evaluate.py

    # or an explicit step
    python 03_evaluate.py --checkpoint outputs/smolvla_lora_grasp/checkpoints/005000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

REPO_ID = "hyzhang01/GCA_parallel_adaptation_b01"
DEFAULT_CHECKPOINT = Path("outputs/smolvla_lora_grasp/checkpoints/last")

# Matches the 40/10 split used by --dataset.eval_split=0.2.
# Fixed indices (not random) so re-runs are comparable across checkpoints.
HELD_OUT_EPISODES = list(range(40, 50))


def resolve_checkpoint_dir(path: Path) -> Path:
    """Accept checkpoints/last, checkpoints/005000, or .../pretrained_model."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if (path / "adapter_model.safetensors").exists() or (path / "model.safetensors").exists():
        return path
    if (path / "pretrained_model").is_dir():
        return path / "pretrained_model"
    raise FileNotFoundError(
        f"No model weights found under {path}. Expected pretrained_model/ or adapter_model.safetensors."
    )


def load_policy(ckpt_dir: Path, device: str):
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.device = device
    cfg.pretrained_path = str(ckpt_dir)

    peft_cfg = PeftConfig.from_pretrained(ckpt_dir)
    # Build with fine-tune feature shapes, load base hub weights, then attach LoRA adapter.
    policy = SmolVLAPolicy.from_pretrained(peft_cfg.base_model_name_or_path, config=cfg)
    policy = PeftModel.from_pretrained(policy, str(ckpt_dir), config=peft_cfg, is_trainable=False)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def episode_local_starts(dataset: LeRobotDataset, episode_indices: list[int]) -> dict[int, int]:
    """Map original episode index -> local frame offset inside a filtered dataset."""
    starts: dict[int, int] = {}
    cursor = 0
    for ep_idx in episode_indices:
        starts[ep_idx] = cursor
        cursor += int(dataset.meta.episodes[ep_idx]["length"])
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint dir (checkpoints/last, checkpoints/005000, or .../pretrained_model)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir = resolve_checkpoint_dir(args.checkpoint)
    print(f"Using checkpoint: {ckpt_dir}")

    print(f"Loading held-out episodes {HELD_OUT_EPISODES[0]}-{HELD_OUT_EPISODES[-1]} from {REPO_ID} ...")
    dataset = LeRobotDataset(REPO_ID, episodes=HELD_OUT_EPISODES)
    starts = episode_local_starts(dataset, HELD_OUT_EPISODES)

    print("Loading policy + processors ...")
    policy, preprocessor, postprocessor = load_policy(ckpt_dir, args.device)

    total_abs_error = 0.0
    total_frames = 0
    per_episode_error: dict[int, float] = {}

    with torch.no_grad():
        for ep_idx in HELD_OUT_EPISODES:
            episode_frames = int(dataset.meta.episodes[ep_idx]["length"])
            start = starts[ep_idx]

            # PeftModel forwards unknown attrs to the base SmolVLAPolicy.
            policy.reset()
            ep_error = 0.0

            for frame_offset in range(episode_frames):
                sample = dataset[start + frame_offset]
                batch = preprocessor(sample)
                pred = policy.select_action(batch)
                pred = postprocessor(pred)

                gt_action = sample["action"].to(pred.device)
                abs_error = (pred.squeeze(0) - gt_action).abs().mean().item()
                ep_error += abs_error
                total_abs_error += abs_error
                total_frames += 1

            per_episode_error[ep_idx] = ep_error / episode_frames
            print(f"Episode {ep_idx}: mean abs action error = {per_episode_error[ep_idx]:.4f}")

    print(
        f"\nOverall mean abs action error across {total_frames} frames: "
        f"{total_abs_error / total_frames:.4f}"
    )
    print(
        "\nInterpretation: this is a raw regression error, not a task success rate. "
        "Use it to compare checkpoints/configs relative to each other (e.g. RUN A "
        "vs RUN B from 02_train_lora.sh), not as an absolute pass/fail threshold. "
        "A real success-rate number requires closed-loop rollout on the arm or in sim."
    )


if __name__ == "__main__":
    main()
