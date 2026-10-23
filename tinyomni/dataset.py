"""On-demand synthetic data with explicit information boundaries.

The 18 worlds are color (3) x shape (3) x horizontal direction (2).
Images reveal color/shape, tones encode color/shape, and videos also reveal
direction. These artificial tones are NOT speech. Targets are canonical
renderings, not noisy copies of the observation. A local generator makes
independent training and validation seeds reproducible without global RNG state.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

ID_TO_TOKEN = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<task_text>", "<task_image>", "<task_audio>",
    "<task_video>", "<task_mixed>",
    "red", "green", "blue", "circle", "square", "triangle", "left", "right",
]
VOCAB = {token: index for index, token in enumerate(ID_TO_TOKEN)}
TASKS = ["text", "image", "audio", "video", "mixed"]
COLORS = ["red", "green", "blue"]
SHAPES = ["circle", "square", "triangle"]
MOTIONS = ["left", "right"]
AUDIO_LENGTH = 256
IMAGE_SIZE = 16


class _RandomSource:
    """Keep old training streams; use full-width NumPy seeds for validation.

    PyTorch's CPU generator can alias seeds differing only in high bits.
    Therefore setting the high bit on a torch seed is NOT a domain separator.
    Low-domain draws retain their original torch call order and dtypes; the
    high domain uses default_rng's SeedSequence/PCG64 with the complete uint64.
    Separate streams do not imply different semantic labels in an 18-world set.
    """

    def __init__(self, seed: int):
        seed = int(seed)
        if not 0 <= seed < 2**64:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        self.numpy = np.random.default_rng(seed) if seed >= 2**63 else None
        self.torch = None if self.numpy is not None else torch.Generator().manual_seed(seed)

    def integers(self, low: int, high: int, size: tuple) -> torch.Tensor:
        if self.numpy is None:
            return torch.randint(low, high, size, generator=self.torch)
        return torch.from_numpy(np.asarray(self.numpy.integers(low, high, size=size,
                                                               dtype=np.int64)))

    def uniform(self, *size: int) -> torch.Tensor:
        if self.numpy is None:
            return torch.rand(*size, generator=self.torch)
        return torch.from_numpy(self.numpy.random(size=size, dtype=np.float32))

    def normal(self, size: tuple) -> torch.Tensor:
        if self.numpy is None:
            return torch.randn(size, generator=self.torch)
        return torch.from_numpy(self.numpy.standard_normal(size=size, dtype=np.float32))


def _render(colors: torch.Tensor, shapes: torch.Tensor,
            centers: torch.Tensor) -> torch.Tensor:
    """Render N RGB images; centers are pixel coordinates [N,2] in x,y order."""
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    dx = xx.float()[None] - centers[:, 0, None, None]
    dy = yy.float()[None] - centers[:, 1, None, None]
    circle = dx.square() + dy.square() <= 3.3 ** 2
    square = (dx.abs() <= 3.1) & (dy.abs() <= 3.1)
    triangle = (dy >= -3.4) & (dy <= 3.4) & (dx.abs() <= (dy + 3.4) * 0.55)
    mask = torch.where(shapes[:, None, None] == 0, circle,
                       torch.where(shapes[:, None, None] == 1, square, triangle))
    palette = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]])
    foreground = palette[colors, :, None, None]
    return torch.where(mask[:, None], foreground, torch.tensor(0.02))


def _tone(colors: torch.Tensor, shapes: torch.Tensor,
          phase: torch.Tensor) -> torch.Tensor:
    """3/6/9 cycles encode color; amplitude .35/.60/.85 encodes shape."""
    time = torch.arange(AUDIO_LENGTH, dtype=torch.float32) / AUDIO_LENGTH
    cycles = torch.tensor([3.0, 6.0, 9.0])[colors]
    amplitude = torch.tensor([0.35, 0.60, 0.85])[shapes]
    return amplitude[:, None] * torch.sin(
        2 * math.pi * cycles[:, None] * time[None] + phase[:, None])


def make_batch(batch_size: int, seed: int, tasks: Iterable[str] | str | None = None,
               device: str | torch.device = "cpu") -> dict:
    """Return a reproducible batch; sensor tasks never receive attribute labels.

    Text is [BOS, task, color-or-UNK, shape-or-UNK, motion-or-UNK]. Only
    the text task receives attribute words. Other tasks must inspect sensors.
    Missing sensor tensors are zeroed AND presence is passed to attention masks.
    Modality ablations may subsequently zero presence without editing tensors.
    Seeds below 2**63 preserve the original torch training stream. Seeds with
    the top bit set use a distinct full-width NumPy generator for validation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chosen = list(TASKS if tasks is None else ([tasks] if isinstance(tasks, str) else tasks))
    if not chosen or any(task not in TASKS for task in chosen):
        raise ValueError(f"tasks must be a nonempty subset of {TASKS}")
    random = _RandomSource(seed)
    colors = random.integers(0, 3, (batch_size,))
    shapes = random.integers(0, 3, (batch_size,))
    motions = random.integers(0, 2, (batch_size,))
    task_choices = random.integers(0, len(chosen), (batch_size,))
    names = [chosen[index] for index in task_choices.tolist()]
    presence = torch.zeros(batch_size, 3, dtype=torch.bool)
    for index, name in enumerate(names):
        if name in ("image", "audio", "video"):
            presence[index, ("image", "audio", "video").index(name)] = True
        elif name == "mixed":
            bits = int(random.integers(1, 8, ()))
            presence[index] = torch.tensor([bool(bits & (1 << axis)) for axis in range(3)])

    color_tokens = colors + VOCAB["red"]
    shape_tokens = shapes + VOCAB["circle"]
    motion_tokens = motions + VOCAB["left"]
    text = torch.full((batch_size, 5), VOCAB["<unk>"], dtype=torch.long)
    text[:, 0] = VOCAB["<bos>"]
    text[:, 1] = torch.tensor([VOCAB[f"<task_{name}>"] for name in names])
    text_task = torch.tensor([name == "text" for name in names])
    text[text_task, 2] = color_tokens[text_task]
    text[text_task, 3] = shape_tokens[text_task]
    text[text_task, 4] = motion_tokens[text_task]
    know_motion = text_task | presence[:, 2]
    targets = torch.stack([
        color_tokens, shape_tokens,
        torch.where(know_motion, motion_tokens, VOCAB["<unk>"]),
        torch.full((batch_size,), VOCAB["<eos>"], dtype=torch.long),
    ], dim=1)

    canonical_centers = torch.full((batch_size, 2), 7.5)
    target_image = _render(colors, shapes, canonical_centers)
    target_audio = _tone(colors, shapes, torch.zeros(batch_size))
    direction = motions.float() * 2 - 1
    target_action = torch.zeros(batch_size, 4, 2)
    target_action[:, :, 0] = direction[:, None] * torch.tensor([0.25, 0.5, 0.75, 1.0])
    # An unknown action is not retained even in the nominal target tensor.
    target_action[~know_motion] = 0

    centers = canonical_centers + 2 * random.uniform(batch_size, 2) - 1
    image = _render(colors, shapes, centers)
    image = (image + 0.015 * random.normal(image.shape)).clamp(0, 1)
    phase = 2 * math.pi * random.uniform(batch_size)
    audio = _tone(colors, shapes, phase)
    audio = audio + 0.015 * random.normal(audio.shape)
    video_centers = centers[:, None, :].expand(-1, 4, -1).clone()
    video_centers[:, :, 0] += direction[:, None] * torch.linspace(-2, 2, 4)
    video = _render(colors[:, None].expand(-1, 4).reshape(-1),
                    shapes[:, None].expand(-1, 4).reshape(-1),
                    video_centers.reshape(-1, 2)).reshape(batch_size, 4, 3, 16, 16)
    video = (video + 0.015 * random.normal(video.shape)).clamp(0, 1)
    image *= presence[:, 0, None, None, None]
    audio *= presence[:, 1, None]
    video *= presence[:, 2, None, None, None, None]
    batch = {
        "text": text, "image": image, "audio": audio, "video": video,
        "presence": presence, "targets": targets,
        "target_image": target_image, "target_audio": target_audio,
        "target_action": target_action,
        "image_loss_mask": torch.ones(batch_size, dtype=torch.bool),
        "audio_loss_mask": torch.ones(batch_size, dtype=torch.bool),
        "action_loss_mask": know_motion, "tasks": names,
    }
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}
