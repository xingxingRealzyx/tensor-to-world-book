"""TinyOmni: from-scratch learning in a finite synthetic multimodal world.

This is a teaching model, not a pretrained assistant or a natural speech model.
"""

from .dataset import ID_TO_TOKEN, TASKS, VOCAB, make_batch
from .model import TinyOmni

__all__ = ["ID_TO_TOKEN", "TASKS", "VOCAB", "make_batch", "TinyOmni"]
