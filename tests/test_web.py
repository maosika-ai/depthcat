"""The local web UI's JSON API, end to end on a random port with the fake backend:
upload → job → poll → download, plus the page itself and the error paths."""

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from reshot.web import make_server

os.environ["RESHOT_FAKE_BACKEND"] = "1"


def _clip(path: Path, frames=12, w=200, h=120, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), (i * 10) % 256, np.uint8))
    vw.release()
    return path


@pytest.fixture
def server(tmp_path):
    srv = make_server(tmp_path / "out", port=0, tmp_dir=tmp_path / "tmp")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv
    srv.shutdown()
    srv.server_close()


def _multipart(field, filename, data, ctype="video/mp4"):
    boundary = "----reshottest"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _post(url, body, ctype):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read(), dict(r.headers)


def test_page_and_info(server):
    base, _ = server
    st, body, _ = _get(base + "/")
    assert st == 200 and b"ReShot" in body and b"maosika.com" in body
    st, body, _ = _get(base + "/api/info")
    info = json.loads(body)
    assert info["targets"]["seedance"]["fps"] == 24 and info["out_dir"].endswith("out")


def test_upload_job_download(server, tmp_path):
    base, _ = server
    clip = _clip(tmp_path / "ref clip.mp4", frames=30)
    body, ctype = _multipart("file", "ref clip.mp4", clip.read_bytes())
    st, src = _post(base + "/api/upload", body, ctype)
    assert st == 200 and src["width"] == 200 and src["frames"] == 30 and src["seconds"] == 1.0
    st, raw, _ = _get(base + src["url"])
    assert st == 200 and len(raw) == clip.stat().st_size

    st, j = _post(
        base + "/api/jobs",
        json.dumps({"source_id": src["id"], "target": "h3", "max_side": 160}).encode(),
        "application/json",
    )
    assert st == 200
    for _ in range(100):
        _, body, _ = _get(base + f"/api/jobs/{j['id']}")
        job = json.loads(body)
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    assert job["result"]["fps"] == 24 and job["result"]["width"] % 32 == 0 and job["result"]["width"] <= 160
    assert any(s["tag"] == "wrote" for s in job["steps"])
    out = Path(job["result"]["output"])
    assert out.exists() and out.parent == tmp_path / "out" and out.name.startswith("ref clip_depth_h3")
    st, raw, hdr = _get(base + job["result"]["output_url"] + "?download=1")
    assert st == 200 and len(raw) == out.stat().st_size and "attachment" in hdr["Content-Disposition"]
    # a second run must not overwrite the first
    st, j2 = _post(
        base + "/api/jobs", json.dumps({"source_id": src["id"], "target": "h3"}).encode(), "application/json"
    )
    for _ in range(100):
        _, body, _ = _get(base + f"/api/jobs/{j2['id']}")
        if json.loads(body)["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert Path(json.loads(body)["result"]["output"]).name == "ref clip_depth_h3_2.mp4"


def test_range_request_for_video_seeking(server, tmp_path):
    base, _ = server
    clip = _clip(tmp_path / "c.mp4")
    body, ctype = _multipart("file", "c.mp4", clip.read_bytes())
    _, src = _post(base + "/api/upload", body, ctype)
    req = urllib.request.Request(base + src["url"], headers={"Range": "bytes=0-99"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 206 and len(r.read()) == 100 and r.headers["Content-Range"].startswith("bytes 0-99/")


def test_bad_inputs(server):
    base, _ = server
    body, ctype = _multipart("file", "x.mp4", b"not a video")
    st, j = _post(base + "/api/upload", body, ctype)
    assert st == 400 and "error" in j
    st, j = _post(base + "/api/jobs", json.dumps({"source_id": "nope"}).encode(), "application/json")
    assert st == 400
    st, j = _post(base + "/api/jobs", b"{", "application/json")
    assert st == 400
