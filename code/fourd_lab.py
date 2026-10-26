"""Train and deploy a channel projection while keeping all four tensor axes.

Training inputs [B,3,H,W] are fresh synthetic arrays. Only the target builder
knows the reference map; the model starts with random weights and learns it.
Prediction loads saved parameters and consumes only the user's input tensor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

AXES = ["B", "C", "H_img", "W_img"]


def reference(x):
    """Target generation only; the prediction command never calls this."""
    weight = x.new_tensor([[1., -2., .5], [-.5, 1., 2.]])
    bias = x.new_tensor([.25, -.75])
    return torch.einsum("bchw,oc->bohw", x, weight) + bias[None, :, None, None]


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("format") != "fourd-channel-v1":
        raise ValueError("Expected a fourd-channel-v1 checkpoint")
    model = nn.Conv2d(3, 2, kernel_size=1).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()


def train(args, device):
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    valid_rng = torch.Generator().manual_seed(args.seed + 100001)
    model = nn.Conv2d(3, 2, kernel_size=1).to(device)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    # Validation uses a separate stream and different image dimensions.
    valid_x = torch.randn(64, 3, 5, 7, generator=valid_rng).to(device)
    valid_y = reference(valid_x)
    with torch.no_grad():
        initial_mse = F.mse_loss(model(valid_x), valid_y).item()
    history = []
    for step in range(1, args.steps + 1):
        x = torch.randn(32, 3, 4, 5, generator=train_rng).to(device)
        target = reference(x)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; update cancelled")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Missing/non-finite gradient; update cancelled")
        optimizer.step()
        history.append({"step": step, "mse": loss.item()})

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save({"format": "fourd-channel-v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "steps": args.steps,
                "seed": args.seed, "input_axes": AXES}, checkpoint)
    restored = load_model(checkpoint, device)
    with torch.inference_mode():
        final_prediction = model(valid_x)
        final_mse = F.mse_loss(final_prediction, valid_y).item()
        reload_error = (restored(valid_x) - final_prediction).abs().max().item()
        # Same learned 1x1 map expressed as a channel-last matrix multiply.
        matrix_form = (valid_x.permute(0, 2, 3, 1)
                       @ restored.weight[:, :, 0, 0].T + restored.bias).permute(0, 3, 1, 2)
        equivalent_error = (matrix_form - restored(valid_x)).abs().max().item()
        # Changing sample 1 must leave sample 0 unchanged.
        altered = valid_x.clone()
        altered[1] += 100
        isolated_error = (restored(altered)[0] - restored(valid_x)[0]).abs().max().item()
        changed = any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
    assert changed and reload_error < 1e-6 and isolated_error == 0
    assert equivalent_error < 1e-5
    report = {
        "format": "fourd-channel-v1", "seed": args.seed, "steps": args.steps,
        "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
        "torch": str(torch.__version__), "train_shape": [32, 3, 4, 5],
        "validation_shape": [64, 3, 5, 7], "output_channels": 2,
        "initial_validation_mse": initial_mse, "final_validation_mse": final_mse,
        "parameter_update": changed, "checkpoint_reload_max_error": reload_error,
        "conv_vs_permuted_matmul_max_error": equivalent_error,
        "cross_sample_interference_max_error": isolated_error,
        "learned_weight": restored.weight[:, :, 0, 0].detach().cpu().tolist(),
        "learned_bias": restored.bias.detach().cpu().tolist(),
        "history": history,
        "scope": "Learning an affine channel map on synthetic 4D arrays; not image recognition.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


def predict(args, device):
    payload = json.loads(Path(args.input).read_text())
    if payload.get("axes", AXES) != AXES:
        raise ValueError("Input axes must be B,C,H_img,W_img")
    x = torch.tensor(payload["images"], dtype=torch.float32, device=device)
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape) < 1:
        raise ValueError("Input must have shape [B,3,H_img,W_img] with positive dimensions")
    if not torch.isfinite(x).all():
        raise ValueError("All input values must be finite")
    model = load_model(args.checkpoint, device)
    with torch.inference_mode():
        y = model(x)
    result = {"input_shape": list(x.shape), "output_shape": list(y.shape),
              "output_axes": ["B", "C_out", "H_img", "W_img"],
              "values": y.cpu().tolist()}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--steps", type=int, default=400)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--out", default="runs/fourd-channels")
    inference = sub.add_parser("predict")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output")
    for command in (training, inference):
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.command == "train" and args.steps < 1:
        parser.error("--steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if args.command == "train":
        train(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
