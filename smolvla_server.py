"""
SmolVLA WebSocket inference server (run in the LeRobot / Farzam env — NOT env_isaaclab).

Keeps a persistent connection: client sends obs messages, server replies with action chunks.
No new HTTP request per query.

Defaults: last checkpoint, port 6002, device cuda, host 0.0.0.0

Terminal A (Farzam / lerobot env) — short form:
    cd /home/fyp/code/Farzam/files && ./serve.sh

Terminal B (env_isaaclab):
    python client/Smolvla_Client.py --host localhost --port 6002 --task_id b01 --num_demos 1

Wire format:
  Preferred: msgpack binary frames (numpy arrays packed natively — no base64).
  Fallback:  JSON text frames (legacy / Smolvla_Client without msgpack).

Messages:
  client -> {"type": "reset"}
  client -> {"type": "infer", "image": ..., "wrist_image": ..., "state": [...], "task": "..."}
  server -> {"type": "metadata", ...}   (on connect)
  server -> {"type": "ok"}
  server -> {"type": "actions", "actions": [[...], ...], "server_timing": {...}}
  server -> {"type": "error", "error": "..."}
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import logging
import os
import pathlib
import time
import traceback
from typing import Any

import msgpack
import numpy as np
import torch
from websockets.sync.server import serve

logger = logging.getLogger("smolvla_server")


# ---------------------------------------------------------------------------
# msgpack + numpy (OpenPI-compatible enough for ndarray round-trips)
# ---------------------------------------------------------------------------

def _pack_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError(f"Cannot msgpack-encode type {type(obj)}")


def _unpack_hook(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__") is True:
        return (
            np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
            .reshape(obj["shape"])
            .copy()
        )
    return obj


def _encode_message(payload: dict[str, Any], *, binary: bool) -> bytes | str:
    if binary:
        return msgpack.packb(payload, default=_pack_default, use_bin_type=True)
    return json.dumps(payload)


def _decode_message(raw: bytes | str) -> tuple[dict[str, Any], bool]:
    """Return (message, used_binary). Binary = msgpack; text = JSON."""
    if isinstance(raw, bytes):
        # Heuristic: JSON text frames from some clients arrive as utf-8 bytes.
        if raw[:1] in (b"{", b"["):
            return json.loads(raw.decode("utf-8")), False
        return msgpack.unpackb(raw, object_hook=_unpack_hook, raw=False), True
    return json.loads(raw), False


def _resolve_model_path(model_path: str) -> str:
    path = pathlib.Path(model_path).expanduser()
    if path.is_dir():
        nested = path / "pretrained_model"
        if nested.is_dir() and (nested / "config.json").exists():
            return str(nested)
        if (path / "config.json").exists():
            return str(path)
    return str(model_path)


def _is_peft_checkpoint(model_path: str) -> bool:
    root = pathlib.Path(model_path)
    return (root / "adapter_config.json").is_file() or (root / "adapter_model.safetensors").is_file()


def _hwc_to_chw_float(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    # Copy: base64/frombuffer arrays are often read-only; torch.from_numpy warns otherwise.
    tensor = torch.from_numpy(np.ascontiguousarray(image).copy()).to(dtype=torch.float32) / 255.0
    return tensor.permute(2, 0, 1).contiguous()


def _decode_image(value: Any) -> np.ndarray:
    """Accept ndarray, nested list, or packed {shape,dtype,data_b64} payloads."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict) and "data_b64" in value:
        raw = base64.b64decode(value["data_b64"])
        arr = np.frombuffer(raw, dtype=np.dtype(value.get("dtype", "uint8")))
        return arr.reshape(value["shape"]).copy()
    if isinstance(value, dict) and "data" in value:
        arr = np.asarray(value["data"], dtype=np.dtype(value.get("dtype", "uint8")))
        if "shape" in value:
            arr = arr.reshape(value["shape"])
        return arr
    return np.asarray(value)


class SmolVLAService:
    """Loads SmolVLA (+ optional LoRA) once and serves action chunks."""

    def __init__(self, model_path: str, device: str = "cuda"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model_path = _resolve_model_path(model_path)
        logger.info("Loading config from %s", self.model_path)

        config = PreTrainedConfig.from_pretrained(self.model_path)
        config.device = str(self.device)

        use_peft = bool(getattr(config, "use_peft", False)) or _is_peft_checkpoint(self.model_path)

        if use_peft:
            from peft import PeftConfig, PeftModel

            logger.info("PEFT/LoRA checkpoint detected (adapter_config.json / use_peft)")
            peft_config = PeftConfig.from_pretrained(self.model_path)
            base_path = peft_config.base_model_name_or_path
            if not base_path:
                raise ValueError("PEFT adapter has no base_model_name_or_path")
            logger.info("Base model: %s", base_path)
            try:
                policy = SmolVLAPolicy.from_pretrained(base_path, config=config, local_files_only=True)
            except Exception as e:
                logger.warning("local_files_only failed (%s); retrying online", e)
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                policy = SmolVLAPolicy.from_pretrained(base_path, config=config)
            policy = PeftModel.from_pretrained(
                policy, self.model_path, config=peft_config, is_trainable=False
            )
        else:
            try:
                policy = SmolVLAPolicy.from_pretrained(self.model_path, config=config, local_files_only=True)
            except Exception as e:
                logger.warning("local_files_only failed (%s); retrying online", e)
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                policy = SmolVLAPolicy.from_pretrained(self.model_path, config=config)

        policy.to(self.device)
        policy.eval()
        self.policy = policy
        self.config = config
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=self.model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        logger.info(
            "Ready on %s | chunk_size=%s | action_dim=%s | peft=%s",
            self.device,
            getattr(config, "chunk_size", "?"),
            config.output_features["action"].shape[0],
            use_peft,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "metadata",
            "ok": True,
            "model_path": self.model_path,
            "device": str(self.device),
            "chunk_size": getattr(self.config, "chunk_size", None),
            "wire": ["msgpack", "json"],
        }

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        elif hasattr(self.policy, "get_base_model"):
            base = self.policy.get_base_model()
            if hasattr(base, "reset"):
                base.reset()

    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        image = _decode_image(payload["image"])
        wrist_image = _decode_image(payload["wrist_image"])
        state = np.asarray(payload["state"], dtype=np.float32).reshape(-1).copy()
        task = payload.get("task") or payload.get("instruction") or ""
        if state.shape[0] != 8:
            raise ValueError(f"Expected state shape (8,), got {state.shape}")

        obs = {
            "observation.images.image": _hwc_to_chw_float(image),
            "observation.images.wrist_image": _hwc_to_chw_float(wrist_image),
            "observation.state": torch.from_numpy(state),
            "task": task,
        }
        batch = self.preprocessor(obs)
        action_chunk = self.policy.predict_action_chunk(batch)
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Expected (B,T,D) actions, got {tuple(action_chunk.shape)}")

        actions = []
        for t in range(action_chunk.shape[1]):
            action_t = self.postprocessor(action_chunk[:, t, :])
            actions.append(action_t.squeeze(0).detach().cpu().numpy().astype(np.float32))
        # JSON clients get lists; msgpack clients get ndarrays (see _encode_message).
        return {"type": "actions", "actions": actions}


def _handle_client(websocket, service: SmolVLAService) -> None:
    peer = getattr(websocket, "remote_address", None)
    logger.info("Client connected: %s", peer)
    # Start in JSON so unknown clients always understand metadata; switch per-request.
    reply_binary = False
    websocket.send(_encode_message(service.metadata(), binary=False))
    try:
        for raw in websocket:
            try:
                msg, reply_binary = _decode_message(raw)
            except Exception as e:
                websocket.send(
                    _encode_message({"type": "error", "error": f"invalid message: {e}"}, binary=False)
                )
                continue

            msg_type = msg.get("type", "infer")
            try:
                if msg_type == "reset":
                    service.reset()
                    websocket.send(_encode_message({"type": "ok"}, binary=reply_binary))
                elif msg_type in ("infer", "act"):
                    t0 = time.time()
                    out = service.predict(msg)
                    out["server_timing"] = {"infer_ms": 1000.0 * (time.time() - t0)}
                    if not reply_binary:
                        out["actions"] = [a.tolist() for a in out["actions"]]
                    websocket.send(_encode_message(out, binary=reply_binary))
                elif msg_type == "ping":
                    websocket.send(_encode_message({"type": "pong"}, binary=reply_binary))
                else:
                    websocket.send(
                        _encode_message(
                            {"type": "error", "error": f"unknown type: {msg_type}"},
                            binary=reply_binary,
                        )
                    )
            except Exception as e:
                logger.error("Handler failed:\n%s", traceback.format_exc())
                websocket.send(
                    _encode_message({"type": "error", "error": str(e)}, binary=reply_binary)
                )
    finally:
        logger.info("Client disconnected: %s", peer)


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA WebSocket inference server")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/fyp/code/Farzam/files/outputs/smolvla_lora_grasp/checkpoints/last",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6002)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    service = SmolVLAService(model_path=args.model_path, device=args.device)

    logger.info("Listening on ws://%s:%d  (messages: reset / infer)", args.host, args.port)

    with serve(
        functools.partial(_handle_client, service=service),
        args.host,
        args.port,
        max_size=64 * 1024 * 1024,
    ) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down")


if __name__ == "__main__":
    main()
