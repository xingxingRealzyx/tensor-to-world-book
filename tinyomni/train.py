"""Train TinyOmni from random weights on a deliberately finite synthetic world."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .dataset import ID_TO_TOKEN, TASKS, VOCAB, make_batch
from .model import TinyOmni

LOW_SEED_MASK = (1 << 63) - 1
VALIDATION_SEED_BIT = 1 << 63
SEED_SCHEME = "uint64-domain-v1: train high bit 0, validation high bit 1"


def training_seed(seed: int, step: int) -> int:
    """Preserve the original seed+step stream until the low 63 bits wrap."""
    return (seed + step) & LOW_SEED_MASK


def validation_seed(seed: int, offset: int = 0) -> int:
    return VALIDATION_SEED_BIT | ((seed + offset) & LOW_SEED_MASK)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or auto")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu or auto")
    return torch.device(name)


def parse_tasks(value: str) -> list[str]:
    tasks = [part.strip() for part in value.split(",") if part.strip()]
    if not tasks or len(tasks) != len(set(tasks)) or any(t not in TASKS for t in tasks):
        raise argparse.ArgumentTypeError(f"tasks must be unique names from {','.join(TASKS)}")
    return tasks


def decoder_inputs(targets: torch.Tensor) -> torch.Tensor:
    bos = torch.full_like(targets[:, :1], VOCAB["<bos>"])
    return torch.cat((bos, targets[:, :-1]), dim=1)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_sample = (prediction - target).square().flatten(1).mean(1)
    weights = mask.to(per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1)


def compute_loss(model: TinyOmni, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(batch, decoder_inputs(batch["targets"]))
    text_loss = F.cross_entropy(
        outputs["logits"].reshape(-1, len(VOCAB)),
        batch["targets"].reshape(-1), ignore_index=VOCAB["<pad>"],
    )
    losses = {"text": text_loss}
    for key in ("image", "audio", "action"):
        losses[key] = masked_mse(outputs[key], batch[f"target_{key}"], batch[f"{key}_loss_mask"])
    total = sum(losses.values())
    if not bool(torch.isfinite(total).item()):
        raise FloatingPointError("non-finite loss; no optimizer update was performed")
    values = {key: float(value.detach().cpu()) for key, value in losses.items()}
    values["total"] = float(total.detach().cpu())
    return total, values


def train_step(model: TinyOmni, optimizer: torch.optim.Optimizer, batch: dict,
               grad_clip: float = 1.0) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, values = compute_loss(model, batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    values["grad_norm"] = float(grad_norm.detach().cpu())
    return values


def generated_metrics(model: TinyOmni, task: str, seed: int, samples: int,
                      batch_size: int, device: torch.device, masked: str | None = None) -> dict:
    """Evaluate actual autoregressive generation, never teacher-forced argmax."""
    token_correct = token_count = exact = count = 0
    mse_sums = {key: 0.0 for key in ("image", "audio", "action")}
    mse_counts = {key: 0 for key in mse_sums}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, samples, batch_size):
            size = min(batch_size, samples - offset)
            batch = make_batch(size, validation_seed(seed, offset), tasks=[task], device=str(device))
            if masked:
                batch[masked] = torch.zeros_like(batch[masked])
                batch["presence"] = batch["presence"].clone()
                batch["presence"][:, {"image": 0, "audio": 1, "video": 2}[masked]] = False
            # Targets and continuous labels must not be visible to inference.
            inputs = {key: value for key, value in batch.items()
                      if key != "targets" and not key.startswith("target_") and not key.endswith("_loss_mask")}
            generated = model.generate(inputs, max_new_tokens=batch["targets"].shape[1])
            truth = batch["targets"]
            predicted = generated["tokens"]
            if predicted.shape[1] < truth.shape[1]:
                predicted = F.pad(predicted, (0, truth.shape[1] - predicted.shape[1]), value=VOCAB["<pad>"])
            predicted = predicted[:, :truth.shape[1]]
            valid = truth != VOCAB["<pad>"]
            correct = predicted == truth
            token_correct += int((correct & valid).sum().item())
            token_count += int(valid.sum().item())
            exact += int((correct | ~valid).all(1).sum().item())
            count += size
            for key in mse_sums:
                errors = (generated[key] - batch[f"target_{key}"]).square().flatten(1).mean(1)
                mask = batch[f"{key}_loss_mask"].bool()
                mse_sums[key] += float(errors[mask].sum().cpu())
                mse_counts[key] += int(mask.sum().item())
    return {
        "samples": count, "token_count": token_count,
        "token_accuracy": token_correct / token_count if token_count else None,
        "exact_match": exact / count if count else None,
        **{f"{key}_mse": mse_sums[key] / mse_counts[key] if mse_counts[key] else None for key in mse_sums},
        **{f"{key}_samples": mse_counts[key] for key in mse_sums},
        "masked_modality": masked,
    }


def evaluate(model: TinyOmni, tasks: list[str], seed: int, samples: int,
             batch_size: int, device: torch.device, diagnose: bool = True) -> dict:
    result = {"generation": {}, "ablations": {}, "seed_scheme": SEED_SCHEME,
              "validation_base_seed": validation_seed(seed)}
    for index, task in enumerate(tasks):
        result["generation"][task] = generated_metrics(
            model, task, validation_seed(seed, index * 10000), samples, batch_size, device,
        )
    if diagnose:
        for index, task in enumerate(("image", "audio")):
            data_seed = validation_seed(seed, 80000 + index * 10000)
            normal = generated_metrics(model, task, data_seed, samples, batch_size, device)
            muted = generated_metrics(model, task, data_seed, samples, batch_size, device, masked=task)
            result["ablations"][task] = {
                "normal": normal, "masked": muted,
                "token_accuracy_drop": normal["token_accuracy"] - muted["token_accuracy"],
                "exact_match_drop": normal["exact_match"] - muted["exact_match"],
                "note": "identical synthetic examples; zero modality tensor and set its presence=False",
            }
    return result


def model_config(model: TinyOmni) -> dict:
    from dataclasses import asdict, is_dataclass
    config = model.config
    return asdict(config) if is_dataclass(config) else dict(config)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, model: TinyOmni, optimizer: torch.optim.Optimizer,
                    step: int, run: dict, history: list[dict]) -> None:
    payload = {
        "format_version": 1, "config": model_config(model),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "step": step, "run": run,
        "history": history, "vocab": list(ID_TO_TOKEN), "torch_rng_state": torch.get_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> tuple[TinyOmni, dict]:
    # Tensor/dict-only checkpoints are read by the restricted PyTorch loader.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("vocab") != list(ID_TO_TOKEN):
        raise ValueError("checkpoint vocabulary differs from this TinyOmni package")
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = TinyOmni(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def write_loss_curve(path: Path, history: list[dict]) -> None:
    """A dependency-free SVG of the actual optimizer-step losses."""
    if not history:
        return
    width, height, margin = 900, 360, 55
    max_step = max(1, history[-1]["step"])
    maximum = max(1e-6, max(row["total"] for row in history))
    points = " ".join(
        f"{margin + row['step']/max_step*(width-2*margin):.2f},"
        f"{height-margin-row['total']/maximum*(height-2*margin):.2f}"
        for row in history
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="55" y="25" font-family="sans-serif" font-size="17">TinyOmni: actual training loss (CE + three masked MSE terms)</text>
<path d="M{margin},{margin} V{height-margin} H{width-margin}" fill="none" stroke="#444"/>
<polyline points="{points}" fill="none" stroke="#176f86" stroke-width="1.5"/>
<text x="8" y="60" font-family="sans-serif" font-size="12">{maximum:.3f}</text>
<text x="30" y="{height-margin}" font-family="sans-serif" font-size="12">0</text>
<text x="{width-margin-80}" y="{height-15}" font-family="sans-serif" font-size="12">step {max_step}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=parse_tasks, default=list(TASKS))
    parser.add_argument("--steps", type=int, default=500, help="total desired optimizer steps, including resumed steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("output/tinyomni"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-samples", type=int, default=128, help="examples per task and ablation")
    parser.add_argument("--eval-every", type=int, default=0, help="0 evaluates only initial and final models")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--threads", type=int, default=2, help="CPU intra-op threads")
    parser.add_argument("--resume", type=Path, help="resume weights AND AdamW state; saved hyperparameters take precedence")
    parser.add_argument("--evaluate-only", action="store_true", help="load --resume and rewrite metrics using the isolated validation seed domain; do not train or rewrite weights")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.eval_samples, args.log_every, args.threads) < 1:
        parser.error("steps, batch-size, eval-samples, log-every and threads must be positive")
    if args.eval_every < 0 or args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        parser.error("invalid optimizer or evaluation settings")
    if args.evaluate_only and not args.resume:
        parser.error("--evaluate-only requires --resume CHECKPOINT")
    if not 0 <= args.seed <= LOW_SEED_MASK:
        parser.error("--seed must be in [0,2**63-1]")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only:
        model, checkpoint = load_checkpoint(args.resume, device)
        started = time.monotonic()
        final = evaluate(model, args.tasks, args.seed, args.eval_samples, args.batch_size, device)
        metrics_path = args.out / "metrics.json"
        previous = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        run = dict(checkpoint["run"])
        run.update({"evaluation_seed": args.seed, "validation_seed_scheme": SEED_SCHEME,
                    "validation_seed_domain": "[2**63,2**64)", "evaluation_device": str(device),
                    "evaluation_tasks": args.tasks, "evaluation_samples_per_task": args.eval_samples,
                    "data_split": "training checkpoint unchanged; fresh validation seeds in high uint64 domain"})
        metrics = {"run": run, "history": checkpoint.get("history", []),
                   "evaluations": [{"step": checkpoint["step"], **final}], "final": final,
                   "evaluation": "autoregressive generate; pad ignored, unk counted; isolated validation seed domain",
                   "reevaluated_checkpoint": str(args.resume.resolve()),
                   "reevaluation_seconds": time.monotonic() - started}
        if previous:
            prior_scheme = previous.get("run", {}).get("seed_scheme", previous.get("run", {}).get("data_split", "unrecorded legacy seed scheme"))
            prior_records = [{**row, "seed_scheme": row.get("seed_scheme", prior_scheme)}
                             for row in previous.get("evaluations", [])]
            metrics["prior_evaluations"] = previous.get("prior_evaluations", []) + prior_records
            if "elapsed_seconds" in previous:
                metrics["elapsed_seconds"] = previous["elapsed_seconds"]
        write_json(metrics_path, metrics)
        write_json(args.out / "config.json", {"model": model_config(model), "run": run, "vocab": list(ID_TO_TOKEN)})
        write_loss_curve(args.out / "loss-curve.svg", checkpoint.get("history", []))
        print(json.dumps({"event": "reevaluated", "out": str(args.out.resolve()),
                          "weights_unchanged": True, "final": final}, ensure_ascii=False), flush=True)
        return
    history: list[dict] = []
    start_step = 0
    model = TinyOmni().to(device)
    checkpoint = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        start_step = int(checkpoint["step"])
        history = list(checkpoint.get("history", []))
        if args.steps < start_step:
            parser.error(f"--steps must be at least the resumed step ({start_step})")
        if checkpoint["run"]["seed"] != args.seed or checkpoint["run"]["tasks"] != args.tasks:
            parser.error("resume requires the original --seed and --tasks; use a new run for a changed data schedule")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if checkpoint:
        if "optimizer" not in checkpoint:
            parser.error("checkpoint has no optimizer state; this command does not silently reset it")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
    run = {
        "seed": args.seed, "tasks": args.tasks, "batch_size": args.batch_size,
        "desired_steps": args.steps, "device": str(device), "threads": args.threads,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"], "grad_clip": args.grad_clip,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "data_split": "training seeds (seed+step)&(2**63-1); validation seeds 2**63|((seed+offset)&(2**63-1))",
        "seed_scheme": SEED_SCHEME, "training_seed_domain": "[0,2**63)",
        "validation_seed_domain": "[2**63,2**64)",
        "scope": "finite synthetic combinations; held-out random perturbations, not unseen-combination or real-world generalization",
        "resume_from": str(args.resume) if args.resume else None,
        "optimizer_restored": bool(checkpoint),
    }
    write_json(args.out / "config.json", {"model": model_config(model), "run": run, "vocab": list(ID_TO_TOKEN)})
    evaluation_seed = args.seed
    print(json.dumps({"event": "start", **run}, ensure_ascii=False), flush=True)
    initial = evaluate(model, args.tasks, evaluation_seed, args.eval_samples, args.batch_size, device, diagnose=False)
    evaluations = [{"step": start_step, **initial}]
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        batch = make_batch(args.batch_size, training_seed(args.seed, step), tasks=args.tasks, device=str(device))
        values = train_step(model, optimizer, batch, args.grad_clip)
        row = {"step": step, **values}
        history.append(row)
        if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"event": "train", **row}, ensure_ascii=False), flush=True)
        if args.eval_every and step % args.eval_every == 0 and step < args.steps:
            snapshot = evaluate(model, args.tasks, evaluation_seed, args.eval_samples, args.batch_size, device, diagnose=False)
            evaluations.append({"step": step, **snapshot})
            print(json.dumps({"event": "validation", "step": step, **snapshot}, ensure_ascii=False), flush=True)
            save_checkpoint(args.out / "checkpoint.pt", model, optimizer, step, run, history)
    final = evaluate(model, args.tasks, evaluation_seed, args.eval_samples, args.batch_size, device, diagnose=True)
    evaluations.append({"step": args.steps, **final})
    metrics = {"run": run, "history": history, "evaluations": evaluations,
               "final": final, "elapsed_seconds": time.monotonic() - started,
               "evaluation": "autoregressive generate; pad ignored, unk counted as a target; MSE only on valid masks"}
    save_checkpoint(args.out / "checkpoint.pt", model, optimizer, args.steps, run, history)
    write_json(args.out / "metrics.json", metrics)
    write_loss_curve(args.out / "loss-curve.svg", history)
    print(json.dumps({"event": "done", "out": str(args.out.resolve()), "final": final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
