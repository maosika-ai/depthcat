"""Screenshot the web page for the README — drives headless Chrome over the DevTools protocol.

Why not Playwright: it downloads its own browser; Chrome is already installed and
`websockets` is enough to talk to it. Usage (server already running on --url):

    python scripts/shoot_web_ui.py --video 走廊武打.mp4 --out docs/img/web_ui.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


async def shoot(url: str, video: Path, out: Path, width: int, height: int, lang: str) -> None:
    port = 9333
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={port}",
            "--no-first-run",
            f"--window-size={width},{height}",
            "--hide-scrollbars",
            f"--user-data-dir=/tmp/reshot-shoot-profile",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except Exception:
                time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=2**30) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(msg["error"])
                        return msg["result"]

            async def js(expr):
                r = await call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                return r["result"].get("value")

            await call(
                "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=False
            )
            await call("Page.enable")
            await call("DOM.enable")
            await call("Page.navigate", url=url)
            await asyncio.sleep(1.0)
            print("page:", await js("location.href + ' | ' + document.title + ' | ' + document.readyState"))
            if lang == "en":
                await js("document.getElementById('lang').click()")
            obj = None
            for _ in range(50):  # the page must have settled before the input exists
                r = await call("Runtime.evaluate", expression="document.getElementById('file')")
                if r["result"].get("objectId"):
                    obj = r["result"]["objectId"]
                    break
                await asyncio.sleep(0.2)
            await call("DOM.setFileInputFiles", files=[str(video.resolve())], objectId=obj)
            for _ in range(100):
                if await js("!document.getElementById('players').hidden"):
                    break
                await asyncio.sleep(0.2)
            await js("document.getElementById('run').click()")
            for _ in range(600):
                if await js("!document.getElementById('result').hidden"):
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(1.5)  # let both players show a frame
            await js("document.querySelectorAll('video').forEach(v => { v.pause(); v.currentTime = 2.0; })")
            await asyncio.sleep(1.0)
            shot = await call("Page.captureScreenshot", format="jpeg", quality=88, captureBeyondViewport=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(shot["data"]))
            print("wrote", out, out.stat().st_size, "bytes")
    finally:
        proc.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765/")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=1360)
    ap.add_argument("--height", type=int, default=980)
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    a = ap.parse_args()
    asyncio.run(shoot(a.url, a.video, a.out, a.width, a.height, a.lang))


if __name__ == "__main__":
    main()
