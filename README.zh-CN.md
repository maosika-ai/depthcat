<h1 align="center">depthcat</h1>
<p align="center"><b>复制走位，不复制演员。</b></p>
<p align="center">把参考视频抽成视频模型能跟着走的深度白模——走位一样、镜头一样，人是你的，画风是你的。</p>

<p align="center"><img src="docs/demo.gif" width="560" alt="左：参考片；右：depthcat 抽出的深度白模"></p>

<p align="center">
<a href="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-blue" alt="python">
<img src="https://img.shields.io/badge/%E8%BF%90%E8%A1%8C%E4%BA%8E-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>
<p align="center"><a href="README.md">English</a> · <a href="docs/USAGE.zh-CN.md">使用手册</a> · <a href="https://huggingface.co/spaces/maosika/depthcat">Hugging Face</a> · <a href="CHANGELOG.md">更新日志</a></p>

---

## 你遇到的问题

你找到一段走位、运镜都对的片子——女主从门口走进来，男人在后面挣扎，镜头缓缓推近。
现在想让视频模型照这个拍：

- **提示词管不了走位。**"女主在前景、男人在她身后、镜头慢推"，写一百遍，每次出来都不一样。
- **直接喂参考片，带走的太多。** 人脸、服装、光线、画风全跟过来；真人素材还大概率过不了模型的审核。

这就是 LibTV「深度动作捕捉」在做的事——把参考片抽成深度白模再去生成。depthcat
把这一步做成了你自己电脑上的一条命令：开源、可商用、能进你自己的流水线。

## depthcat 做什么

把参考片剥到只剩你真正想要的那一层：**谁在哪、多大、怎么动、镜头怎么动**——
输出近白远黑的深度视频，正是深度 ControlNet 训练时吃的格式。

```bash
pip install git+https://github.com/maosika-ai/depthcat
depthcat 参考片.mp4 -o 白模.mp4 --target h3
```

```
参考片.mp4 ──▶ depthcat ──▶ 白模.mp4 ──┐
                                        ├──▶ 视频模型（深度 ControlNet）──▶ 你的片子
你的提示词：人物、服装、光线、画风 ─────┘
```

把 `白模.mp4` 交给 MiniMax H3 Fun ControlNet、Wan VACE 或任何带深度控制的生成模型，
提示词只写**长相**，出来的片子保留参考片的走位和镜头，不带一点它的身份。

## 给谁用

- **短剧、影视团队** —— 把一段好走位套到自己的角色身上。
- **广告、产品视频** —— 把参考广告的运镜放到自己的产品上。
- **反复打磨一个场景的人** —— 走位锁一次，人物和画风随便换。

## 为什么不用别的

| 替代方案 | 你要放弃什么 |
|---|---|
| 某个视频 App 里的深度捕捉 | 只能在那个 App 里用；没法写脚本、批处理、接进自己的流水线。 |
| ComfyUI 里逐帧跑单图深度节点 | 会闪——每帧各自归一化。fps、尺寸、编码还得自己搭。 |
| 托管的深度视频 API | 按秒收费，素材要传到别人服务器。测一下行，当流水线不行。 |
| **depthcat** | 一条命令、一个 Python 函数。视频原生模型、整段归一化、生成模型预设、端到端 Apache-2.0。8 GB 显卡或 Mac 就能跑。 |

## 决定成败的细节

- **时序一致性来自模型，不是靠模糊。** Video Depth Anything 用 32 帧窗口重叠推理并对齐；单图模型做不到。
- **整段视频只归一化一次。** 逐帧 min/max 会让静止的墙在有人走近镜头时忽明忽暗，生成模型把它读成运动。
- **按模型分辨率推理，按你要的尺寸输出。** 12 秒 720p 只要约 4 GB 内存，不是 8–20 GB。跑之前先估算，放不下就拒绝。
- **帧按时间戳选、尺寸靠裁、质量靠 CRF。** 30→24 fps 真的是 24；尺寸裁到生成模型要的倍数、绝不补边；x264 CRF 12，控制信号没有色带。
- **能随产品一起发的许可。** 默认 Apache-2.0 权重；非商用的变体要显式指定，运行时会明说。

## 预设

| `--target` | fps | 尺寸 | 上限 | 对应 |
|---|---|---|---|---|
| `h3` | 24 | 32 的倍数 | 15 秒 | [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) |
| `wan` | 16 | 16 的倍数 | – | Wan 2.1 VACE（按文档取值，尚未端到端实测） |
| `none` | 原片 | 偶数 | – | 通用 |

## 实测

| 设备 | 精度 | ms / 帧 | 宿主内存 |
|---|---|---|---|
| RTX 4090，torch 2.8 cu128 | fp16 | **70**（`--input-size 364` 时 31） | 4.5 GB |
| Apple M2 Max，torch 2.9.1 | fp32 | 约 500 | 约 4 GB |

294 帧 @ 736×1280，Small 模型。更多平台与细节见[使用手册](docs/USAGE.zh-CN.md)。

安装时国内请先 `export HF_ENDPOINT=https://hf-mirror.com`，权重 111 MB。

## Python

```python
from pathlib import Path
from depthcat import RunConfig, run

result = run(RunConfig(input=Path("参考片.mp4"), output=Path("白模.mp4"), target="h3"))
print(result.frames, result.ms_per_frame)
```

## 许可

代码 Apache-2.0。默认权重 Video-Depth-Anything-**Small**，Apache-2.0
（[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)，
CVPR 2025；模型代码 vendored 在 `depthcat/third_party/`，见 [NOTICE](NOTICE)）。
Base/Large 权重是 CC-BY-NC-4.0，只在显式指定时加载。

## 路线图

`--people-only`（SAM 2 人物剪影）· 同一遍推理顺带出骨架 / 边缘 / 法线 ·
PyPI · Docker · ComfyUI 示例工作流 · 可在线运行的 Hugging Face Space

## 参与

欢迎 issue 和 PR —— [CONTRIBUTING.md](CONTRIBUTING.md)。CI 在 Ubuntu 和 Windows 上跑 `ruff check` 和 `pytest`，外加 CPU 真模型。

<p align="center"><sub>出品：<a href="https://www.maosika.com">猫斯卡</a>，每天都在出 AI 短剧的工作室。</sub></p>
