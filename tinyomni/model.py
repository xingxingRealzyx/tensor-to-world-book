"""A small encoder/causal-decoder trained from scratch, with no pretrained parts.

All modalities enter the same encoder. Key-padding masks exclude absent
modalities in every encoder layer and in decoder cross-attention. Continuous
heads read the encoder CLS vector, never the teacher-forced answer tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from .dataset import VOCAB


class TinyOmni(nn.Module):
    DEFAULT_CONFIG = {
        "d_model": 64, "nhead": 4, "dim_feedforward": 128,
        "num_encoder_layers": 2, "num_decoder_layers": 2,
        "dropout": 0.0, "patch_size": 8, "audio_frame_size": 32,
        "max_decode_len": 8, "vocab_size": len(VOCAB),
    }

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            unknown = set(config) - set(self.config)
            if unknown:
                raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
            self.config.update(config)
        cfg = self.config
        width, patch, frame = cfg["d_model"], cfg["patch_size"], cfg["audio_frame_size"]
        if width % cfg["nhead"] or 16 % patch or 256 % frame:
            raise ValueError("nhead must divide d_model; patches/frames must tile their inputs")
        if cfg["vocab_size"] != len(VOCAB):
            raise ValueError("This teaching model uses the dataset's fixed vocabulary")
        self.num_patches = (16 // patch) ** 2
        self.num_audio_frames = 256 // frame
        self.token_embedding = nn.Embedding(len(VOCAB), width, padding_idx=VOCAB["<pad>"])
        self.type_embedding = nn.Embedding(4, width)  # text, image, tone, video
        self.image_projection = nn.Conv2d(3, width, patch, stride=patch)
        self.video_projection = nn.Conv2d(3, width, patch, stride=patch)
        self.audio_projection = nn.Linear(frame, width)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.text_position = nn.Parameter(torch.empty(1, 5, width))
        self.patch_position = nn.Parameter(torch.empty(1, self.num_patches, width))
        self.audio_position = nn.Parameter(torch.empty(1, self.num_audio_frames, width))
        self.video_time = nn.Parameter(torch.empty(1, 4, 1, width))
        self.decoder_position = nn.Parameter(torch.empty(1, cfg["max_decode_len"], width))
        encoder_layer = nn.TransformerEncoderLayer(
            width, cfg["nhead"], cfg["dim_feedforward"], cfg["dropout"],
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, cfg["num_encoder_layers"], norm=nn.LayerNorm(width),
            enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            width, cfg["nhead"], cfg["dim_feedforward"], cfg["dropout"],
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, cfg["num_decoder_layers"], norm=nn.LayerNorm(width))
        self.language_head = nn.Linear(width, len(VOCAB))
        self.image_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                        nn.Linear(128, 3 * 16 * 16), nn.Sigmoid())
        self.audio_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 256))
        self.action_head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 8))
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        for parameter in (self.cls_token, self.text_position, self.patch_position,
                          self.audio_position, self.video_time, self.decoder_position):
            nn.init.normal_(parameter, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[VOCAB["<pad>"]].zero_()

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        text, presence = batch["text"], batch["presence"].bool()
        size = text.shape[0]
        if text.shape != (size, 5) or presence.shape != (size, 3):
            raise ValueError("Expected text[B,5] and presence[B,3]")
        text_tokens = self.token_embedding(text) + self.text_position + self.type_embedding.weight[0]
        image_tokens = self.image_projection(batch["image"]).flatten(2).transpose(1, 2)
        image_tokens = image_tokens + self.patch_position + self.type_embedding.weight[1]
        audio_frames = batch["audio"].unfold(-1, self.config["audio_frame_size"],
                                               self.config["audio_frame_size"])
        audio_tokens = self.audio_projection(audio_frames) + self.audio_position + self.type_embedding.weight[2]
        video = batch["video"].reshape(size * 4, 3, 16, 16)
        video_tokens = self.video_projection(video).flatten(2).transpose(1, 2)
        video_tokens = video_tokens.reshape(size, 4, self.num_patches, -1)
        video_tokens = (video_tokens + self.patch_position[:, None]
                        + self.video_time + self.type_embedding.weight[3]).flatten(1, 2)
        source = torch.cat([self.cls_token.expand(size, -1, -1), text_tokens,
                            image_tokens, audio_tokens, video_tokens], dim=1)
        # PyTorch key_padding_mask: True means this KEY cannot be read.
        # CLS is always valid, preventing an all-masked attention row.
        ignored = torch.cat([
            torch.zeros(size, 1, dtype=torch.bool, device=text.device),
            text.eq(VOCAB["<pad>"]),
            (~presence[:, 0, None]).expand(-1, self.num_patches),
            (~presence[:, 1, None]).expand(-1, self.num_audio_frames),
            (~presence[:, 2, None]).expand(-1, 4 * self.num_patches),
        ], dim=1)
        memory = self.encoder(source, src_key_padding_mask=ignored)
        return memory, ignored

    def _decode(self, ids: torch.Tensor, memory: torch.Tensor,
                ignored: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if not 1 <= length <= self.config["max_decode_len"]:
            raise ValueError("decoder length is outside configured limits")
        embedded = self.token_embedding(ids) + self.decoder_position[:, :length]
        # True ABOVE the diagonal forbids reading future answer positions.
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                      device=ids.device), diagonal=1)
        hidden = self.decoder(
            embedded, memory, tgt_mask=causal,
            tgt_key_padding_mask=ids.eq(VOCAB["<pad>"]),
            memory_key_padding_mask=ignored)
        return self.language_head(hidden)

    def _continuous(self, memory: torch.Tensor) -> dict:
        condition = memory[:, 0]  # no decoder activations, no targets
        return {"image": self.image_head(condition).reshape(-1, 3, 16, 16),
                "audio": self.audio_head(condition),
                "action": self.action_head(condition).reshape(-1, 4, 2)}

    def forward(self, batch: dict, decoder_input_ids: torch.Tensor) -> dict:
        memory, ignored = self._encode(batch)
        result = self._continuous(memory)
        result["logits"] = self._decode(decoder_input_ids, memory, ignored)
        return result

    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int = 4) -> dict:
        """Greedy fixed-length generation. EOS is followed by PAD, never labels."""
        if not 1 <= max_new_tokens <= self.config["max_decode_len"]:
            raise ValueError("max_new_tokens is outside configured limits")
        was_training = self.training
        self.eval()
        try:
            memory, ignored = self._encode(batch)
            ids = torch.full((memory.shape[0], 1), VOCAB["<bos>"],
                             dtype=torch.long, device=memory.device)
            finished = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
            new_tokens = []
            for _ in range(max_new_tokens):
                token = self._decode(ids, memory, ignored)[:, -1].argmax(-1)
                token = torch.where(finished, VOCAB["<pad>"], token)
                new_tokens.append(token)
                finished |= token.eq(VOCAB["<eos>"])
                ids = torch.cat([ids, token[:, None]], dim=1)
            result = self._continuous(memory)
            result["tokens"] = torch.stack(new_tokens, dim=1)
            return result
        finally:
            self.train(was_training)
