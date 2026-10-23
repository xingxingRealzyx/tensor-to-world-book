"""Serve a real TinyOmni checkpoint on localhost; only synthetic task/seed requests."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading

import torch

from .dataset import TASKS
from .infer import response_payload
from .train import choose_device, load_checkpoint


def build_server(checkpoint_path: Path | str, port: int = 8765,
                 device_name: str = "auto", request_timeout: float = 5.0) -> ThreadingHTTPServer:
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    device = choose_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            self.request.settimeout(request_timeout)
            super().setup()

        def reply(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.reply(404, {"error": "use GET /health or POST /infer"})
                return
            self.reply(200, {"status": "ok", "model": "TinyOmni", "training_step": checkpoint["step"],
                             "device": str(device), "tasks": list(TASKS), "scope": "synthetic task/seed only"})

        def do_POST(self) -> None:
            if self.path != "/infer":
                self.reply(404, {"error": "unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    self.reply(413, {"error": "JSON request must be 1..4096 bytes"})
                    return
                if self.headers.get_content_type() != "application/json":
                    self.reply(415, {"error": "Content-Type must be application/json"})
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                payload = json.loads(body)
                if not isinstance(payload, dict) or set(payload) != {"task", "seed"}:
                    raise ValueError("request must contain exactly task and seed; file uploads are not supported")
                task, seed = payload["task"], payload["seed"]
                if not isinstance(task, str) or task not in TASKS:
                    raise ValueError(f"task must be one of {list(TASKS)}")
                if type(seed) is not int or not 0 <= seed < 2**31:
                    raise ValueError("seed must be an integer in [0,2147483647]")
            except TimeoutError:
                self.close_connection = True
                self.reply(408, {"error": "request body read timed out"})
                return
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            try:
                with inference_lock:
                    result = response_payload(model, task, seed, device)
                result["training_step"] = checkpoint["step"]
                self.reply(200, result)
            except Exception as exc:
                self.reply(500, {"error": f"model inference failed: {type(exc).__name__}"})

        def log_message(self, fmt, *args) -> None:
            print("localhost " + fmt % args, flush=True)

    # Health/body parsing stay responsive; one inference at a time protects model state.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--read-timeout", type=float, default=5.0, help="socket request read timeout in seconds")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("port must be 1..65535 and threads must be positive")
    if not math.isfinite(args.read_timeout) or args.read_timeout <= 0:
        parser.error("--read-timeout must be finite and positive")
    torch.set_num_threads(args.threads)
    server = build_server(args.checkpoint, args.port, args.device, args.read_timeout)
    print(f"TinyOmni listening on http://127.0.0.1:{server.server_port}; synthetic examples only", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
