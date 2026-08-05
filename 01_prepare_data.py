"""
Stage 0: Data audit for hyzhang01/GCA_parallel_adaptation_b01

Run this BEFORE any training. It answers the three questions that decide
how the rest of the pipeline is configured:

  1. Is there real per-episode language, or one fixed instruction?
  2. Are the camera keys what SmolVLA expects (observation.images.*)?
  3. What do the action/state stats look like — any outlier episodes
     that would skew LoRA's small-sample training?

Prerequisite (lerobot >= 0.4 / == 0.6.0):
    1) Convert Hub v2.1 dataset to v3.0 once locally:

        python -m lerobot.scripts.convert_dataset_v21_to_v30 \\
            --repo-id=hyzhang01/GCA_parallel_adaptation_b01 \\
            --push-to-hub=false

    2) Rename raw feature keys to LeRobot convention:

        python 00_rename_features.py

Usage:
    python 01_prepare_data.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "hyzhang01/GCA_parallel_adaptation_b01"

# After 00_rename_features.py, keys match LeRobot convention.
# Keep fallbacks for a raw Hub download that was not renamed yet.
SMOLVLA_EXPECTED = {
    "state": "observation.state",
    "action": "action",
    "cameras": "observation.images.*",
}


def _pick_feature(features: dict, *candidates: str) -> str | None:
    for key in candidates:
        if key in features:
            return key
    return None


def _task_strings(meta) -> list[str]:
    """Normalize task metadata across lerobot versions / dataset layouts."""
    tasks = meta.tasks
    if tasks is None:
        return []

    # v3: pandas DataFrame indexed by task string
    if hasattr(tasks, "index") and hasattr(tasks, "columns"):
        return [str(t) for t in tasks.index.tolist()]

    # older: mapping / list-like
    if isinstance(tasks, dict):
        if "task" in tasks:
            return [str(t) for t in tasks["task"]]
        return [str(t) for t in tasks.keys()]

    try:
        return [str(t) for t in list(tasks)]
    except TypeError:
        return []


def main():
    print(f"Loading metadata for {REPO_ID} ...")
    dataset = LeRobotDataset(REPO_ID)
    features = dataset.features

    state_key = _pick_feature(features, "observation.state", "state")
    action_key = _pick_feature(features, "action", "actions")

    # --- 1. Basic shape sanity check ---
    print("\n=== Dataset shape ===")
    print(f"Episodes:      {dataset.num_episodes}")
    print(f"Frames:        {dataset.num_frames}")
    print(f"FPS:           {dataset.fps}")
    print(f"Camera keys:   {list(dataset.meta.camera_keys)}")
    print(f"All features:  {list(features.keys())}")

    if state_key:
        print(f"State shape:   {features[state_key]['shape']}  (key={state_key!r})")
    else:
        print("State shape:   NOT FOUND (looked for observation.state / state)")

    if action_key:
        print(f"Action shape:  {features[action_key]['shape']}  (key={action_key!r})")
    else:
        print("Action shape:  NOT FOUND (looked for action / actions)")

    # Flag key-name mismatches that will break SmolVLA training unless remapped
    print("\n=== SmolVLA key compatibility ===")
    camera_keys = list(dataset.meta.camera_keys)
    smolvla_cams = [k for k in camera_keys if k.startswith("observation.images.")]
    if not smolvla_cams:
        print(
            f"  WARNING: cameras are {camera_keys}, but SmolVLA expects "
            f"{SMOLVLA_EXPECTED['cameras']}. You will need a feature rename / "
            "dataset config map before training (e.g. image→observation.images.front, "
            "wrist_image→observation.images.wrist)."
        )
    else:
        print(f"  Cameras OK: {smolvla_cams}")

    if state_key != "observation.state":
        print(
            f"  WARNING: state key is {state_key!r}, SmolVLA expects "
            f"{SMOLVLA_EXPECTED['state']!r}."
        )
    if action_key != "action":
        print(
            f"  WARNING: action key is {action_key!r}, SmolVLA expects "
            f"{SMOLVLA_EXPECTED['action']!r}."
        )

    # --- 2. Language instruction audit ---
    print("\n=== Language instructions ===")
    tasks = _task_strings(dataset.meta)
    if tasks:
        print(f"Distinct task strings found: {len(tasks)}")
        for t in tasks[:10]:
            print(f"  - {t!r}")
        if len(tasks) == 1:
            print(
                "\n  WARNING: only one task string across all episodes. "
                "Fine-tuning will be effectively single-instruction — "
                "don't expect language generalization to new phrasing/objects."
            )
    else:
        print("  No task metadata found — check dataset.meta.tasks directly.")

    # --- 3. Action / state stat audit ---
    print("\n=== Action stats (used for MEAN_STD normalization) ===")
    action_stats = dataset.meta.stats.get(action_key or "action", {})
    for stat_name in ("mean", "std", "min", "max"):
        if stat_name in action_stats:
            print(f"  {stat_name}: {action_stats[stat_name]}")
    if not action_stats:
        print("  (no action stats found)")

    print("\n=== State stats ===")
    state_stats = dataset.meta.stats.get(state_key or "observation.state", {})
    for stat_name in ("mean", "std", "min", "max"):
        if stat_name in state_stats:
            print(f"  {stat_name}: {state_stats[stat_name]}")
    if not state_stats:
        print("  (no state stats found)")

    # --- 4. Episode length distribution ---
    print("\n=== Episode length distribution ===")
    lengths = []
    for i in range(dataset.num_episodes):
        ep = dataset.meta.episodes[i]
        if "length" in ep:
            lengths.append(int(ep["length"]))
    if lengths:
        mean_len = sum(lengths) / len(lengths)
        print(f"  min={min(lengths)}  max={max(lengths)}  mean={mean_len:.1f}")
        outliers = [length for length in lengths if length > 2 * mean_len]
        if outliers:
            print(
                f"  WARNING: {len(outliers)} episode(s) more than 2x mean length: "
                f"{outliers}"
            )

    print("\nAudit complete. Fix any WARNING above before proceeding to training.")


if __name__ == "__main__":
    main()
