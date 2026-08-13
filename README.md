# SmolVLA + LoRA Grasping Fine-Tune — hyzhang01/GCA_suction_franka_a01_id0

## Strategy recap

- **Model**: SmolVLA-450M — LeRobot-native format matches the dataset, action
  expert already uses flow matching.
- **Method**: LoRA fine-tune, not full fine-tune — 50 episodes is small
  enough that full fine-tuning risks overfitting / catastrophic forgetting
  of pretrained vision-language grounding.
- **Data**: Franka Panda, parallel gripper, dual camera (front + wrist),
  7-dim delta EE action, 8-dim state.

## Correction from the planning phase

Earlier in this project I proposed hand-wiring LoRA as: _adapt the vision
encoder, freeze the language decoder, fully fine-tune the action expert._
That was reasoning from first principles, without checking the actual
library. After inspecting the installed `lerobot==0.6.0` source directly
(`modeling_smolvla.py::_get_default_peft_targets`), the real picture is
different:

- LeRobot ships **built-in PEFT support** (`policy.wrap_with_peft()`) — no
  custom LoRA wiring code needed at all, just CLI flags.
- SmolVLA's **default** LoRA target is the **action expert's attention only**
  (q_proj, v_proj inside `lm_expert`), plus full training of a few small glue
  projections. The **entire VLM (vision + language) stays frozen** by
  default — more conservative than what I'd proposed, and arguably better
  suited to 50 episodes since it never touches pretrained visual/language
  representations at all.

The scripts below use this verified default as the first run, with an
optional override (commented out) to extend LoRA into the vision encoder
if evaluation later shows a wrist-camera domain gap. Don't turn that on
preemptively — see whether the default actually struggles first.

## Pipeline

| Step | File                 | Purpose                                                                                          |
| ---- | -------------------- | ------------------------------------------------------------------------------------------------ |
| 0    | `01_prepare_data.py` | Data audit: language diversity, camera keys, stat outliers, episode length spread                |
| 1    | `requirements.txt`   | Environment                                                                                      |
| 2    | `02_train_lora.sh`   | LoRA fine-tune via `lerobot-train` (RUN A = default targets, RUN B = vision-extended, commented) |
| 3    | `03_evaluate.py`     | Open-loop held-out replay — sanity check before real-arm testing                                 |

## Run order

```bash
pip install -r requirements.txt

# Stage 0: audit before touching any model
python 01_prepare_data.py

# Stage 2: train (RUN A first, always)
bash 02_train_lora.sh

# Stage 5: evaluate on held-out episodes 40-49
python 03_evaluate.py --checkpoint outputs/smolvla_lora_suction_a01/checkpoints/last
```

## Known gaps you'll need to fill in before running for real

1. **`01_prepare_data.py`** assumes `dataset.meta.tasks` / `dataset.meta.episodes[i]["length"]`
   exist in the exact shape used here — LeRobotDataset's metadata API has
   shifted across versions, so if a field name errors out, run
   `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('hyzhang01/GCA_suction_franka_a01_id0'); print(d.meta)"`
   and adjust field names to match what's actually returned.
2. **`--policy.pretrained_path=lerobot/smolvla_base`** — confirm this is
   still the correct published checkpoint repo id on the Hub at the time
   you run this; check https://huggingface.co/lerobot for the current name.
3. **`03_evaluate.py`**'s held-out split (episodes 40-49) assumes episode
   indices are stable and sequential in the dataset — verify against the
   audit output from step 0 before trusting the split.
4. Neither script here has been executed against live GPU + the actual
   dataset (this container has no GPU and no HF Hub network access) — treat
   this as a verified-against-source-code starting point, not a
   run-tested pipeline. Expect to debug one or two API mismatches on
   first run.
5. for running the client (Isaac Lab env). `--task` is the *sim* id; language sent to SmolVLA
   comes from task_id a01 (`pick up the cracker box`), not from `--task`.
   python client/Smolvla_Client.py --host 127.0.0.1 --port 6002 --task Grasp-Franka-Vacuum-IK-Rel-img --task_id a01 --num_demos 1
