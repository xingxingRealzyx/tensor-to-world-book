"""CPU regression tests: python -m tinyomni.tests."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import copy
from io import BytesIO, StringIO
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

import torch

from .dataset import VOCAB, make_batch
from .infer import predict, png_bytes, wav_bytes
from .model import TinyOmni
from .serve import build_server
from .train import (compute_loss, decoder_inputs, generated_metrics,
                    load_checkpoint, save_checkpoint, train_step,
                    training_seed, validation_seed, main as train_main)


class TinyOmniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.model = TinyOmni()
        self.batch = make_batch(4, 901, tasks=["text", "image", "audio", "video", "mixed"])

    def test_optimizer_step_is_real_and_finite(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        before = [p.detach().clone() for p in self.model.parameters()]
        metrics = train_step(self.model, optimizer, self.batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, self.model.parameters())))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
        self.assertTrue(optimizer.state)

    def test_non_finite_loss_prevents_update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        broken = copy.deepcopy(self.batch)
        broken["target_image"][0, 0, 0, 0] = float("nan")
        before = [p.detach().clone() for p in self.model.parameters()]
        with self.assertRaises(FloatingPointError):
            train_step(self.model, optimizer, broken)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, self.model.parameters())))
        self.assertFalse(optimizer.state)

    def test_checkpoint_roundtrip_and_optimizer_state(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        metrics = train_step(self.model, optimizer, self.batch)
        self.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, 1, {"seed": 123}, [{"step": 1, **metrics}])
            restored, checkpoint = load_checkpoint(path)
            restored.eval()
            with torch.inference_mode():
                original = self.model.generate(self.batch, max_new_tokens=4)
                replay = restored.generate(self.batch, max_new_tokens=4)
            for key in original:
                torch.testing.assert_close(original[key], replay[key], rtol=0, atol=0)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.001)
            self.assertTrue(all(float(state["step"]) == 1 for state in restored_optimizer.state.values()))

    def test_future_decoder_input_cannot_change_earlier_logits(self):
        self.model.eval()
        decoder = decoder_inputs(self.batch["targets"])
        changed = decoder.clone()
        changed[:, 3] = (changed[:, 3] + 1) % len(VOCAB)
        with torch.inference_mode():
            first = self.model(self.batch, decoder)["logits"]
            second = self.model(self.batch, changed)["logits"]
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-6)

    def test_heads_and_generation_do_not_read_ground_truth(self):
        self.model.eval()
        clean = {key: value for key, value in self.batch.items()
                 if key != "targets" and not key.startswith("target_") and not key.endswith("_loss_mask")}
        decoder = decoder_inputs(self.batch["targets"])
        other_decoder = torch.full_like(decoder, VOCAB["red"])
        poisoned = copy.deepcopy(self.batch)
        poisoned["targets"] = torch.full_like(poisoned["targets"], VOCAB["blue"])
        for key in ("image", "audio", "action"):
            poisoned[f"target_{key}"] = torch.full_like(poisoned[f"target_{key}"], float("nan"))
        with torch.inference_mode():
            original = self.model(self.batch, decoder)
            isolated = self.model(clean, other_decoder)
            generated = self.model.generate(clean, max_new_tokens=4)
            generated_poisoned = self.model.generate(poisoned, max_new_tokens=4)
        for key in ("image", "audio", "action"):
            torch.testing.assert_close(original[key], isolated[key], rtol=0, atol=0)
        for key in generated:
            torch.testing.assert_close(generated[key], generated_poisoned[key], rtol=0, atol=0)

    def test_invalid_action_targets_are_not_scored(self):
        batch = make_batch(4, 99, tasks=["image"])
        self.assertFalse(bool(batch["action_loss_mask"].any()))
        self.model.eval()
        _, values = compute_loss(self.model, batch)
        self.assertEqual(values["action"], 0.0)
        result = generated_metrics(self.model, "image", 999, 5, 3, torch.device("cpu"))
        self.assertEqual(result["samples"], 5)
        self.assertEqual(result["token_count"], 20)  # UNK and EOS are actual labels.
        self.assertIsNone(result["action_mse"])
        self.assertEqual(result["action_samples"], 0)

    def test_training_validation_seed_domains_and_streams(self):
        boundary = 1 << 63
        for seed in (0, 42, boundary - 1):
            for offset in (0, 1000, 1_000_042, boundary + 7):
                self.assertLess(training_seed(seed, offset), boundary)
                self.assertGreaterEqual(validation_seed(seed, offset), boundary)
                self.assertLess(validation_seed(seed, offset), 1 << 64)
        self.assertEqual(training_seed(42, 1000), 1042)
        original = make_batch(8, 42, tasks=["image"])
        validation = make_batch(8, validation_seed(42), tasks=["image"])
        repeated = make_batch(8, validation_seed(42), tasks=["image"])
        self.assertFalse(torch.equal(original["image"], validation["image"]))
        torch.testing.assert_close(validation["image"], repeated["image"], rtol=0, atol=0)

    def test_evaluate_only_does_not_rewrite_checkpoint(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, 1000, {"seed": 42}, [])
            original_bytes = path.read_bytes()
            with redirect_stdout(StringIO()):
                train_main(["--evaluate-only", "--resume", str(path), "--out", str(root),
                            "--device", "cpu", "--eval-samples", "2", "--batch-size", "2", "--threads", "1"])
            self.assertEqual(original_bytes, path.read_bytes())
            metrics = json.loads((root / "metrics.json").read_text())
            self.assertEqual(metrics["evaluations"][0]["step"], 1000)
            self.assertEqual(metrics["final"]["validation_base_seed"], (1 << 63) + 42)

    def test_exported_media_contains_real_output(self):
        batch, generated, report = predict(self.model, "audio", 91, "cpu")
        image = png_bytes(generated["image"][0])
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        with wave.open(BytesIO(wav_bytes(generated["audio"][0])), "rb") as handle:
            self.assertEqual(handle.getnframes(), 256)
            self.assertEqual(handle.getframerate(), 8000)
        self.assertIsNone(report["predicted_actions"])
        self.assertFalse(report["action_target_is_observable"])
        _, _, observable = predict(self.model, "video", 91, "cpu")
        self.assertEqual(len(observable["predicted_actions"]), 4)

    def test_local_http_health_inference_and_validation(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, 0, {"seed": 123}, [])
            server = build_server(path, port=0, device_name="cpu")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/health", timeout=5) as reply:
                    self.assertEqual(json.load(reply)["status"], "ok")
                request = Request(base + "/infer", data=json.dumps({"task": "image", "seed": 11}).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=5) as reply:
                    result = json.load(reply)
                self.assertEqual(result["task"], "image")
                self.assertTrue(base64.b64decode(result["generated_image_png_base64"]).startswith(b"\x89PNG"))
                invalid = Request(base + "/infer", data=b'{"task":"image","seed":11,"upload":"x"}',
                                  headers={"Content-Type": "application/json"})
                with self.assertRaises(HTTPError) as context:
                    urlopen(invalid, timeout=5)
                self.assertEqual(context.exception.code, 400)
                context.exception.close()
            finally:
                server.shutdown()
                worker.join(timeout=5)
                server.server_close()

    def test_incomplete_http_body_times_out_and_health_recovers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, 0, {"seed": 123}, [])
            server = build_server(path, port=0, device_name="cpu", request_timeout=0.15)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as connection:
                    connection.sendall(
                        b"POST /infer HTTP/1.1\r\nHost: localhost\r\n"
                        b"Content-Type: application/json\r\nContent-Length: 50\r\n\r\n{"
                    )
                    response = connection.recv(4096)
                    self.assertIn(b"408", response.split(b"\r\n", 1)[0])
                with urlopen(f"http://127.0.0.1:{server.server_port}/health", timeout=2) as reply:
                    self.assertEqual(json.load(reply)["status"], "ok")
            finally:
                server.shutdown()
                worker.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
