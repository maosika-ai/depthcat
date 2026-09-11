<p align="center"><img src="docs/demo.gif" width="560" alt="左：原片；右：depthcat 抽出的深度白模"></p>

<h1 align="center">depthcat</h1>

<p align="center"><b>把任意视频抽成深度白模——让视频模型复制一段参考片的走位和运镜，却不带走它的人脸、服装和画风。</b></p>

<p align="center">
<a href="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-blue" alt="python">
<img src="https://img.shields.io/badge/%E8%BF%90%E8%A1%8C%E4%BA%8E-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>

<p align="center"><a href="README.md">English</a> · <a href="docs/USAGE.zh-CN.md">使用手册</a> · <a href="CHANGELOG.md">更新日志</a></p>

---

```bash
pip install git+https://github.com/maosika-ai/depthcat
depthcat 参考片.mp4 -o 白模.mp4 --target h3
```

整个流程就这两行。把 `白模.mp4` 连同你自己的提示词一起交给带深度控制的视频模型
（MiniMax H3 Fun ControlNet、Wan VACE……），生成出来的片子**走位和镜头跟参考片**，
**长相和画风全听你的提示词**。就是 LibTV「深度动作捕捉」那种输出，但你自己掌握、可商用。

## 为什么是深度白模

一段参考视频里有两种信息：**怎么拍的**（谁站哪、多大、怎么动、镜头怎么动）和
**长什么样**（人脸、衣服、光线、风格）。深度图只保留前者、丢掉后者。所以它是

- **最干净的控制信号**——"按这段参考片的拍法拍我的剧本"
- **一层合规隔离**——人脸、服装、品牌一样都不会从参考片带出来
- **模型无关**——所有深度 ControlNet 训练时吃的就是这种图

depthcat 负责把这个信号做对、做稳，把那些"随手跑一下深度模型"必然踩的坑提前处理掉。

## 做对了什么

| | depthcat | 常见的"逐帧跑深度模型"脚本 |
|---|---|---|
| 时序一致性 | 原生视频模型（32 帧窗口、重叠、跨窗口对齐） | 闪烁 |
| 归一化 | 整段视频只做一次 | 逐帧 → 静止的墙在"呼吸" |
| 分辨率 | 按模型分辨率推理，8 位结果逐帧放大 | 每帧存全分辨率浮点深度 → 8–20 GB 内存 |
| 帧率 | 按时间戳选帧（30→24 真的是 24） | 整数步长 → 还是 30 |
| 输出尺寸 | 裁到生成模型要的倍数（H3 是 32） | 补黑边 → 模型当成一堵墙 |
| 编码 | x264 CRF 12，2 秒一个关键帧 | 默认 CRF → 色带 → 出片抖动 |
| 许可 | 默认 Apache-2.0 权重，非商用权重需显式指定并警告 | 硬盘上有啥用啥 |
| 安全 | 跑前估内存，超预算拒跑并给出 `--max-frames` 建议 | 把机器交换到死 |

## 安装

```bash
pip install git+https://github.com/maosika-ai/depthcat      # 需要 PATH 里有 ffmpeg
pip install "depthcat[ffmpeg] @ git+https://github.com/maosika-ai/depthcat"   # 自带 ffmpeg 二进制
```

首次运行自动下载权重（111 MB，Apache-2.0）。国内：`export HF_ENDPOINT=https://hf-mirror.com`。

## 用法

```bash
depthcat in.mp4 -o out.mp4                    # 保持原 fps / 尺寸，近白远黑
depthcat in.mp4 -o out.mp4 --target h3        # MiniMax H3 ControlNet：24 fps、边长 32 的倍数、≤15 秒
depthcat in.mp4 -o out.mp4 --npz d.npz --metrics run.json
```

```python
from pathlib import Path
from depthcat import RunConfig, run

result = run(RunConfig(input=Path("in.mp4"), output=Path("out.mp4"), target="h3"))
print(result.ms_per_frame, result.peak_rss_bytes)
```

全部参数、预设、退出码和实战配方：**[docs/USAGE.zh-CN.md](docs/USAGE.zh-CN.md)**。

## 目标预设

| `--target` | fps | 尺寸 | 上限 | 对应 |
|---|---|---|---|---|
| `none` | 原片 | 偶数 | – | 通用 |
| `h3` | 24 | 32 的倍数 | 15 秒 | [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) |
| `wan` | 16 | 16 的倍数 | – | Wan 2.1 VACE（按文档取值，尚未端到端实测） |

## 实测性能（Small 模型，294 帧 @ 736×1280）

| 设备 | 精度 | ms / 帧 | 宿主内存峰值 |
|---|---|---|---|
| RTX 4090，torch 2.8 cu128 | fp16 | **70**（`--input-size 364` 时 31） | 4.5 GB |
| Apple M2 Max，torch 2.9.1 | fp32 | 约 500 | 约 4 GB |
| Apple M2 Max，torch 2.6.0 | fp32 | 约 1700 | — |
| CPU（M2 Max） | fp32 | 约 1850 | — |

fp16 只在 CUDA 上开：MPS 上慢到不可用。Apple 芯片请把 torch 升到 ≥ 2.9。

## 模型与许可

| 权重 | 许可 | 商用 |
|---|---|---|
| Video-Depth-Anything-**Small**（默认） | Apache-2.0 | ✅ |
| Video-Depth-Anything-Base / -Large | CC-BY-NC-4.0 | ❌ 需 `--model` 显式指定，运行时警告 |

depthcat 自身代码 Apache-2.0。模型代码来自
[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
（CVPR 2025），vendored 在 `depthcat/third_party/`，见 [NOTICE](NOTICE)。

## 路线图

- `--people-only`：只留人物剪影、背景抹平（SAM 2）
- `--also pose,canny,normal`：一遍出 MiniMax H3 Fun ControlNet 的其余输入
- Docker 镜像、PyPI 包、ComfyUI 示例工作流
- Hugging Face Space 在线试用

## 参与

欢迎 issue 和 PR，见 [CONTRIBUTING.md](CONTRIBUTING.md)。推之前跑 `ruff check` 和 `pytest`；CI 还会在 CPU 上真跑一遍模型。

出品：[猫斯卡](https://www.maosika.com)，AI 短剧工作室。
