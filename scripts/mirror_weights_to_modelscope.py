#!/usr/bin/env python3
"""Mirror ReShot's default model weights to ModelScope (modelscope.cn/models/maosika/reshot).

Why: the ModelScope model page is marked "pre-release" until the repo contains weight files, and
users in China download from modelscope.cn far faster than from Hugging Face. The three files are
exactly the ones `reshot` pulls from Hugging Face on first run — same names, same bytes (SHA-256
checked here before anything is uploaded), all Apache-2.0:

    video_depth_anything_vits.pth   depth-anything/Video-Depth-Anything-Small   (ByteDance)
    yolox_l.onnx                    yzd-v/DWPose                                (person detector)
    dw-ll_ucoco_384.onnx            yzd-v/DWPose                                (whole-body pose)

Usage (one command; the token is read from the environment and never written anywhere):

    pip install modelscope huggingface_hub
    export MODELSCOPE_API_TOKEN=<your SDK token from https://modelscope.cn/my/myaccesstoken>
    python scripts/mirror_weights_to_modelscope.py            # download (or reuse cache) → verify → upload
    python scripts/mirror_weights_to_modelscope.py --dry-run  # download + verify only, no upload

Uploaded layout in the ModelScope repo: `weights/<original filename>`. Keep that stable — a future
`RESHOT_WEIGHTS_SOURCE=modelscope` would resolve against these paths.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ID = "maosika/reshot"
DEST_DIR = "weights"

# (hf_repo, filename, sha256). The hashes are the LFS object ids on Hugging Face, i.e. the bytes
# `reshot` itself runs on; a mismatch means a corrupted or substituted download — abort, don't upload.
WEIGHTS = [
    (
        "depth-anything/Video-Depth-Anything-Small",
        "video_depth_anything_vits.pth",
        "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
    ),
    (
        "yzd-v/DWPose",
        "yolox_l.onnx",
        "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411",
    ),
    (
        "yzd-v/DWPose",
        "dw-ll_ucoco_384.onnx",
        "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_and_verify() -> list[tuple[Path, str]]:
    """Return [(local_path, path_in_repo)] for all three files, or exit non-zero."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is missing: pip install huggingface_hub")

    out: list[tuple[Path, str]] = []
    for hf_repo, filename, expected in WEIGHTS:
        print(f"→ {hf_repo}/{filename}", flush=True)
        local = Path(hf_hub_download(hf_repo, filename))  # cache hit if already downloaded
        got = sha256(local)
        if got != expected:
            sys.exit(f"SHA-256 mismatch for {filename}:\n  expected {expected}\n  got      {got}\nNot uploading.")
        print(f"  {local.stat().st_size / 1e6:,.1f} MB  sha256 ok")
        out.append((local, f"{DEST_DIR}/{filename}"))
    return out


def upload(files: list[tuple[Path, str]], token: str) -> None:
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        sys.exit("modelscope is missing: pip install modelscope")

    api = HubApi()
    api.login(token)
    for local, path_in_repo in files:
        print(f"↑ {path_in_repo}", flush=True)
        api.upload_file(
            repo_id=REPO_ID,
            repo_type="model",
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            commit_message=f"weights: mirror {local.name} (same bytes as the Hugging Face original, sha256 verified)",
        )
    print(f"\nDone. https://modelscope.cn/models/{REPO_ID}/files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="download + verify only; do not upload")
    args = ap.parse_args()

    files = fetch_and_verify()
    if args.dry_run:
        print("\n--dry-run: all three files verified, nothing uploaded.")
        return
    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if not token:
        sys.exit(
            "MODELSCOPE_API_TOKEN is not set.\n"
            "Get your SDK token at https://modelscope.cn/my/myaccesstoken, then:\n"
            "  export MODELSCOPE_API_TOKEN=<token>\n"
            "and run this script again. The token is only read from the environment."
        )
    upload(files, token)


if __name__ == "__main__":
    main()
