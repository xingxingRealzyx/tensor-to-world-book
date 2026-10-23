"""Run a saved TinyOmni model on one reproducible synthetic example."""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import struct
import wave
import zlib

import torch

from .dataset import ID_TO_TOKEN, TASKS, VOCAB, make_batch
from .train import choose_device, load_checkpoint, write_json

AUDIO_SAMPLE_RATE = 8000


def token_strings(ids: torch.Tensor) -> list[str]:
    return [ID_TO_TOKEN[int(index)] for index in ids.detach().cpu().tolist()]


def png_bytes(tensor: torch.Tensor) -> bytes:
    """Encode a [3,H,W] floating RGB tensor as PNG using the standard library."""
    pixels = (tensor.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError("PNG expects [3,H,W]")
    _, height, width = pixels.shape
    interleaved = pixels.permute(1, 2, 0).contiguous()
    scanlines = b"".join(b"\x00" + bytes(row.flatten().tolist()) for row in interleaved)
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def wav_bytes(tensor: torch.Tensor, repeats: int = 1) -> bytes:
    samples = (tensor.detach().cpu().flatten().clamp(-1, 1) * 32767).round().to(torch.int16).tolist()
    samples = samples * repeats
    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_SAMPLE_RATE)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return output.getvalue()


def predict(model, task: str, seed: int, device: torch.device | str) -> tuple[dict, dict, dict]:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    batch = make_batch(1, seed, tasks=[task], device=str(device))
    # Ground truth stays outside the model input, including during generation.
    inputs = {key: value for key, value in batch.items()
              if key != "targets" and not key.startswith("target_") and not key.endswith("_loss_mask")}
    model.eval()
    with torch.inference_mode():
        generated = model.generate(inputs, max_new_tokens=4)
    expected_ids = batch["targets"][0]
    predicted_ids = generated["tokens"][0]
    action_available = bool(batch["action_loss_mask"][0].item())
    report = {
        "task": task, "seed": seed,
        "input_text_tokens": token_strings(batch["text"][0]),
        "presence": {key: bool(batch["presence"][0, i].item()) for i, key in enumerate(("image", "audio", "video"))},
        "predicted_tokens": token_strings(predicted_ids),
        "predicted_text": " ".join(token for token in token_strings(predicted_ids)
                                   if token not in ("<pad>", "<bos>", "<eos>")),
        "synthetic_ground_truth_tokens": token_strings(expected_ids),
        "predicted_actions": generated["action"][0].detach().cpu().tolist() if action_available else None,
        "synthetic_ground_truth_actions": batch["target_action"][0].detach().cpu().tolist() if action_available else None,
        "action_target_is_observable": action_available,
        "action_available": action_available,
        "action_note": "supervised 2D teaching trajectory" if action_available else "motion is unobservable; this action head output was not supervised and is not exposed",
        "action_interface": "4-step normalized 2D teaching trajectory; not a robot command",
        "audio": {"samples": 256, "sample_rate_hz_for_export": AUDIO_SAMPLE_RATE,
                  "duration_seconds": 256 / AUDIO_SAMPLE_RATE,
                  "meaning": "synthetic color/shape tone, not natural speech"},
    }
    for key in ("image", "audio", "action"):
        valid = bool(batch[f"{key}_loss_mask"][0].item())
        report[f"{key}_mse"] = float((generated[key] - batch[f"target_{key}"]).square().mean().cpu()) if valid else None
    return batch, generated, report


def response_payload(model, task: str, seed: int, device: torch.device | str) -> dict:
    _, generated, report = predict(model, task, seed, device)
    report["generated_image_png_base64"] = base64.b64encode(png_bytes(generated["image"][0])).decode("ascii")
    report["generated_audio_wav_base64"] = base64.b64encode(wav_bytes(generated["audio"][0])).decode("ascii")
    return report


def export_prediction(out: Path, batch: dict, generated: dict, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("input_image", batch["image"][0]),
                         ("generated_image", generated["image"][0]),
                         ("target_image", batch["target_image"][0])):
        (out / f"{name}.png").write_bytes(png_bytes(tensor))
    for index, frame in enumerate(batch["video"][0]):
        (out / f"input_video_{index:02d}.png").write_bytes(png_bytes(frame))
    for name, tensor in (("input_audio", batch["audio"][0]),
                         ("generated_audio", generated["audio"][0]),
                         ("target_audio", batch["target_audio"][0])):
        (out / f"{name}.wav").write_bytes(wav_bytes(tensor))
        # Repeated playback is labeled explicitly; it is not a longer model prediction.
        (out / f"{name}_repeated_preview.wav").write_bytes(wav_bytes(tensor, repeats=20))
    write_json(out / "prediction.json", report)
    write_json(out / "actions.json", {
        "predicted": report["predicted_actions"],
        "synthetic_ground_truth": report["synthetic_ground_truth_actions"],
        "target_is_observable": report["action_target_is_observable"],
        "interface": report["action_interface"],
    })
    (out / "predicted.txt").write_text(report["predicted_text"] + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="mixed")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", type=Path, default=Path("output/tinyomni-inference"))
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    batch, generated, report = predict(model, args.task, args.seed, device)
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["training_step"] = checkpoint["step"]
    export_prediction(args.out, batch, generated, report)
    print(json.dumps({"out": str(args.out.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
