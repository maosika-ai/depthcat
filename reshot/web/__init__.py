"""The local web UI: `reshot` with no arguments (or `reshot web`) starts it and opens a browser.

One HTML page (``index.html`` next to this file) talks to a small JSON API served by the
standard library — no framework, no extra dependency, so ``pip install reshot`` is all a
user needs. Everything stays on the user's machine: uploads go to a temp folder, results to
``~/ReShot`` (or ``--out``).

API
    POST /api/upload            multipart "file"  → source info (id, size, fps, frames, seconds)
    GET  /api/source/<id>       the uploaded clip, for the preview player
    POST /api/jobs              JSON options      → {"id"}; runs in the single worker thread
                                ``control`` picks depth (default) / pose / canny
    GET  /api/jobs/<id>         status, progress steps, result / error
    GET  /api/output/<id>       the control video (``?download=1`` for an attachment)
    GET  /api/keypoints/<id>    the pose JSON of a pose job (``?download=1`` likewise)
    GET  /api/info              version, device, VRAM, output folder, which controls can run

Why one worker thread with cached models: a model is loaded once per server process
(3–6 s for depth, ~1 s for pose) and every job of that control reuses it; two jobs at once
would fight for the GPU, so they queue. Why polling instead of SSE: 500 ms polls are plenty for a 20-second job and keep
the server a plain ``http.server``.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from email.parser import BytesParser
from email.policy import HTTP
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .._version import __version__
from ..config import CONTROLS, RunConfig
from ..errors import ReshotError
from ..io import probe_video
from ..pipeline import _cuda_total_bytes, _make_backend, _run_one
from ..targets import TARGETS

log = logging.getLogger("reshot.web")

MAX_UPLOAD_BYTES = 500 * 2**20
DEFAULT_PORT = 8765


class _Job:
    """One depth run: options in, progress steps and a result (or error) out."""

    def __init__(self, cfg: RunConfig, source_name: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.source_name = source_name
        self.status = "queued"  # queued | running | done | error
        self.steps: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None

    # Reporter protocol — the pipeline calls this for every stage.
    def step(self, tag: str, message: str) -> None:
        self.steps.append({"tag": tag, "message": message, "t": round(time.time() - (self.started or time.time()), 1)})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "source_name": self.source_name,
            "control": self.cfg.control,
            "target": self.cfg.target,
            "quality": self.cfg.quality,
            "steps": self.steps,
            "result": self.result,
            "error": self.error,
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
        }


class _State:
    """Everything the handlers share. One instance per server."""

    def __init__(self, out_dir: Path, tmp_dir: Path) -> None:
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir
        self.sources: dict[str, dict] = {}
        self.jobs: dict[str, _Job] = {}
        self.queue: queue.Queue[_Job] = queue.Queue()
        self.backends: dict[str, tuple] = {}  # control → (backend, name); each loaded once
        self.lock = threading.Lock()
        self.device = "?"

    @property
    def backend(self):
        """The depth backend, if loaded (kept for callers that predate pose / canny)."""
        return self.backends.get("depth", (None, ""))[0]

    def worker(self) -> None:
        while True:
            job = self.queue.get()
            job.status = "running"
            job.started = time.time()
            try:
                control = job.cfg.control
                if control not in self.backends:
                    if control != "canny":
                        job.step("model", f"loading the {control} model (once per session)")
                    self.backends[control] = _make_backend(job.cfg)
                    if self.backends[control][0] is not None:
                        self.device = self.backends[control][0].device
                backend, backend_name = self.backends[control]
                res = _run_one(job.cfg, job, backend, backend_name)
                job.result = {
                    "control": control,
                    "output": str(res.output),
                    "output_url": f"/api/output/{job.id}",
                    "keypoints_url": f"/api/keypoints/{job.id}" if job.cfg.keypoints else None,
                    "frames": res.frames,
                    "fps": res.fps,
                    "width": res.width,
                    "height": res.height,
                    "device": res.device,
                    "precision": res.precision,
                    "ms_per_frame": round(res.ms_per_frame, 1),
                    "depth_seconds": round(res.depth_seconds, 1),
                    "total_seconds": round(res.total_seconds, 1),
                    "model_input_resolution": res.metrics.get("model_input_resolution"),
                    "gpu_peak_reserved_bytes": res.metrics.get("gpu_peak_reserved_bytes"),
                    "output_bytes": res.metrics.get("output_bytes"),
                    "people_per_frame_max": res.metrics.get("people_per_frame_max"),
                }
                job.status = "done"
            except ReshotError as exc:
                job.error = str(exc)
                job.status = "error"
            except Exception as exc:  # a bug, not a user problem — show it rather than hang
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"
                log.error("job %s crashed:\n%s", job.id, traceback.format_exc())
            finally:
                job.finished = time.time()
                self.queue.task_done()


def _onnxruntime_installed() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "clip"
    return "".join(c if c.isalnum() or c in "-_ ." else "_" for c in stem)[:80].strip() or "clip"


class _Handler(BaseHTTPRequestHandler):
    state: _State  # set by serve()
    index_html: bytes

    # ---- plumbing -------------------------------------------------------------
    def log_message(self, fmt, *args):  # quiet unless -v
        log.debug("%s " + fmt, self.address_string(), *args)

    def _json(self, obj, status=HTTPStatus.OK) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, download_name: str | None = None) -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        # Range support so the <video> element can seek.
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = HTTPStatus.OK
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if b and a else size - 1
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # ---- routes -----------------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if not parts or parts == ["index.html"]:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(self.index_html)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(self.index_html)
            elif parts == ["api", "info"]:
                self._json(self._info())
            elif parts[:2] == ["api", "source"] and len(parts) == 3:
                src = self.state.sources.get(parts[2])
                self._file(Path(src["path"])) if src else self._json({"error": "unknown source"}, HTTPStatus.NOT_FOUND)
            elif parts[:2] == ["api", "jobs"] and len(parts) == 3:
                job = self.state.jobs.get(parts[2])
                self._json(job.to_dict()) if job else self._json({"error": "unknown job"}, HTTPStatus.NOT_FOUND)
            elif parts == ["api", "jobs"]:
                self._json([j.to_dict() for j in sorted(self.state.jobs.values(), key=lambda j: j.created)])
            elif parts[:2] == ["api", "output"] and len(parts) == 3:
                job = self.state.jobs.get(parts[2])
                if not job or not job.result:
                    return self._json({"error": "no output yet"}, HTTPStatus.NOT_FOUND)
                q = parse_qs(url.query)
                self._file(Path(job.result["output"]), Path(job.result["output"]).name if q.get("download") else None)
            elif parts[:2] == ["api", "keypoints"] and len(parts) == 3:
                job = self.state.jobs.get(parts[2])
                if not job or not job.result or not job.cfg.keypoints or not Path(job.cfg.keypoints).exists():
                    return self._json({"error": "no keypoints for this job"}, HTTPStatus.NOT_FOUND)
                q = parse_qs(url.query)
                self._file(Path(job.cfg.keypoints), Path(job.cfg.keypoints).name if q.get("download") else None)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser closed the video stream; normal

    def do_POST(self) -> None:
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if parts == ["api", "upload"]:
            return self._upload()
        if parts == ["api", "jobs"]:
            return self._create_job()
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _info(self) -> dict:
        st = self.state
        vram = _cuda_total_bytes("cuda") if st.device == "cuda" else 0
        return {
            "version": __version__,
            "device": st.device,
            "vram_bytes": vram,
            "out_dir": str(st.out_dir),
            "targets": {
                k: {"fps": v.fps, "multiple": v.multiple, "max_seconds": v.max_seconds} for k, v in TARGETS.items()
            },
            "controls": list(CONTROLS),
            "pose_available": _onnxruntime_installed(),
            "model_loaded": st.backend is not None,
            "models_loaded": sorted(st.backends),
        }

    def _upload(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return self._json(
                {"error": f"upload must be 1 byte – {MAX_UPLOAD_BYTES // 2**20} MB"}, HTTPStatus.BAD_REQUEST
            )
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._json({"error": "expected multipart/form-data"}, HTTPStatus.BAD_REQUEST)
        raw = self.rfile.read(length)
        # The email parser handles multipart/form-data correctly and is in the stdlib.
        msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
        part = next((p for p in msg.iter_parts() if p.get_param("name", header="content-disposition") == "file"), None)
        if part is None:
            return self._json({"error": "no 'file' field"}, HTTPStatus.BAD_REQUEST)
        filename = part.get_filename() or "clip.mp4"
        data = part.get_payload(decode=True)
        sid = uuid.uuid4().hex[:12]
        path = self.state.tmp_dir / f"{sid}_{_safe_stem(filename)}{Path(filename).suffix.lower() or '.mp4'}"
        path.write_bytes(data)
        try:
            w, h, fps, n = probe_video(path)
        except Exception as exc:
            path.unlink(missing_ok=True)
            return self._json({"error": f"cannot read that file as a video ({exc})"}, HTTPStatus.BAD_REQUEST)
        if w <= 0 or h <= 0:
            path.unlink(missing_ok=True)
            return self._json({"error": "that file has no video stream"}, HTTPStatus.BAD_REQUEST)
        info = {
            "id": sid,
            "name": filename,
            "path": str(path),
            "bytes": len(data),
            "width": w,
            "height": h,
            "fps": round(fps, 3),
            "frames": n,
            "seconds": round(n / fps, 2) if fps else 0,
            "url": f"/api/source/{sid}",
        }
        self.state.sources[sid] = info
        self._json({k: v for k, v in info.items() if k != "path"})

    def _create_job(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, HTTPStatus.BAD_REQUEST)
        src = self.state.sources.get(str(body.get("source_id", "")))
        if not src:
            return self._json({"error": "upload a video first"}, HTTPStatus.BAD_REQUEST)
        target = str(body.get("target", "seedance"))
        control = str(body.get("control", "depth"))
        if control == "pose" and not _onnxruntime_installed():
            return self._json(
                {"error": 'pose needs ONNX Runtime: pip install "reshot[pose]", then restart reshot'},
                HTTPStatus.BAD_REQUEST,
            )
        stem = _safe_stem(src["name"])
        out = self.state.out_dir / f"{stem}_{control}_{target}.mp4"
        n = 2
        while out.exists():  # never overwrite an earlier result
            out = self.state.out_dir / f"{stem}_{control}_{target}_{n}.mp4"
            n += 1
        try:
            canny = body.get("canny") or [100, 200]
            cfg = RunConfig(
                input=Path(src["path"]),
                output=out,
                target=target,
                control=control,
                quality=str(body.get("quality", "fast")),
                max_res=int(body.get("max_side") or 1280) if body.get("max_side") else 1280,
                max_frames=int(body["max_frames"]) if body.get("max_frames") else None,
                invert=bool(body.get("invert", False)),
                clip_percent=float(body.get("clip_percent", 0.0) or 0.0),
                gamma=float(body.get("gamma", 1.0) or 1.0),
                metrics=out.with_suffix(".json"),
                force=bool(body.get("force", False)),
                pose_hands=bool(body.get("pose_hands", True)),
                pose_face=bool(body.get("pose_face", False)),
                pose_smooth=bool(body.get("pose_smooth", True)),
                keypoints=out.with_name(out.stem + ".keypoints.json") if control == "pose" else None,
                canny_low=int(canny[0]),
                canny_high=int(canny[1]),
            )
        except (ReshotError, ValueError, TypeError, IndexError) as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        job = _Job(cfg, src["name"])
        self.state.jobs[job.id] = job
        self.state.queue.put(job)
        self._json({"id": job.id})


def _free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return 0


def make_server(out_dir: Path, port: int = 0, tmp_dir: Path | None = None) -> ThreadingHTTPServer:
    """Build (but don't run) the server — used by `serve()` and by the tests."""
    import tempfile

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="reshot-web-"))
    tmp.mkdir(parents=True, exist_ok=True)
    state = _State(out_dir, tmp)
    threading.Thread(target=state.worker, name="reshot-worker", daemon=True).start()

    handler = type("ReShotHandler", (_Handler,), {})
    handler.state = state
    handler.index_html = (Path(__file__).parent / "index.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    server.reshot_state = state  # type: ignore[attr-defined]
    return server


def serve(out_dir: Path | None = None, port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    """Run the UI until Ctrl-C. Returns an exit code."""
    out = Path(out_dir) if out_dir else Path.home() / "ReShot"
    chosen = _free_port(port)
    if not chosen:
        print(f"reshot: no free port between {port} and {port + 19}", file=sys.stderr)
        return 1
    server = make_server(out, chosen)
    url = f"http://127.0.0.1:{chosen}/"
    print(f"ReShot {__version__} — open {url}   (results go to {out};  Ctrl-C to stop)", file=sys.stderr)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nreshot: stopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0


__all__ = ["DEFAULT_PORT", "make_server", "serve"]
