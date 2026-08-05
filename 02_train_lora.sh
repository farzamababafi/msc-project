#!/usr/bin/env bash
# Stage 2: LoRA fine-tune SmolVLA on hyzhang01/GCA_parallel_adaptation_b01
#
# IMPORTANT CORRECTION to the plan discussed earlier in chat:
# There is no custom PEFT wiring to write by hand. lerobot==0.6.0 ships
# `policy.wrap_with_peft()` (see lerobot/policies/pretrained.py) which is
# driven entirely by CLI flags below. `lerobot-train` calls it automatically
# whenever `--peft.*` flags are present.
#
# ALSO IMPORTANT: SmolVLA's *default* PEFT target (if you omit --peft.target_modules)
# is NOT "LoRA the vision encoder, freeze text, full-FT the action expert" as
# I proposed earlier by reasoning alone. The real default, verified from
# modeling_smolvla.py::_get_default_peft_targets(), is:
#
#   - LoRA applied to: the action expert's (lm_expert) q_proj and v_proj only
#   - Fully trained (not LoRA, not frozen): the small glue projections
#     (state_proj, action_in_proj, action_out_proj, action_time_mlp_in/out)
#   - Frozen entirely: the whole VLM (vision encoder AND language decoder)

# If eval later shows the model struggles specifically with the wrist-cam
# view (domain gap from pretraining), extend LoRA into the vision encoder
# with RUN B's --peft.target_modules override — don't reach for it up front.

set -euo pipefail

DATASET_REPO_ID="hyzhang01/GCA_parallel_adaptation_b01"
BASE_CHECKPOINT="lerobot/smolvla_base"   # published SmolVLA-450M pretrained weights
OUTPUT_DIR="outputs/smolvla_lora_grasp"
# Local-only fine-tune; do not push checkpoints to the Hub
POLICY_REPO_ID="local/smolvla_lora_grasp"

# RTX 4080 16GB: batch 8 + 512px SmolVLA often OOMs; 4 is a safer default.
BATCH_SIZE="${BATCH_SIZE:-4}"

# ---------------------------------------------------------------------------
# RUN A (recommended first pass) — LeRobot's default PEFT targets for SmolVLA
# ---------------------------------------------------------------------------
lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.eval_split=0.2 \
  --policy.type=smolvla \
  --policy.pretrained_path="${BASE_CHECKPOINT}" \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.r=16 \
  --peft.lora_alpha=32 \
  --batch_size="${BATCH_SIZE}" \
  --steps=5000 \
  --save_freq=500 \
  --eval_steps=250 \
  --num_workers=4 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=smolvla_lora_grasp \
  --seed=1000

# ---------------------------------------------------------------------------
# RUN B (only if RUN A's eval shows wrist-cam / visual domain gap) —
# extends LoRA into the SigLIP vision encoder's attention projections.
# Uncomment to use. Verify the exact module path with:
#   python -c "from lerobot.policies.factory import make_policy; ..."
#   then print(model) and grep for 'vision_model' to confirm names match
#   your installed transformers version before relying on this regex.
# ---------------------------------------------------------------------------
# lerobot-train \
#   --dataset.repo_id="${DATASET_REPO_ID}" \
#   --dataset.eval_split=0.2 \
#   --policy.type=smolvla \
#   --policy.pretrained_path="${BASE_CHECKPOINT}" \
#   --policy.load_vlm_weights=true \
#   --policy.device=cuda \
#   --policy.repo_id="${POLICY_REPO_ID}_vision" \
#   --policy.push_to_hub=false \
#   --peft.method_type=LORA \
#   --peft.r=16 \
#   --peft.lora_alpha=32 \
#   --peft.target_modules='(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.vlm_with_expert\.vlm\.model\.vision_model\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))' \
#   --batch_size="${BATCH_SIZE}" \
#   --steps=5000 \
#   --save_freq=500 \
#   --eval_steps=250 \
#   --output_dir="${OUTPUT_DIR}_vision_lora" \
#   --job_name=smolvla_lora_grasp_vision \
#   --seed=1000

echo "Training complete. Checkpoints + LoRA adapter weights saved to ${OUTPUT_DIR}"
