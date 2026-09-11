# video2blockout

[![ci](https://github.com/maosika-ai/video2blockout/actions/workflows/ci.yml/badge.svg)](https://github.com/maosika-ai/video2blockout/actions/workflows/ci.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

把任意视频抽成**深度白模视频**——近白远黑的灰度片，喂给视频生成模型的 ControlNet
（MiniMax H3 Fun ControlNet、Wan VACE……），只复制参考片的**走位与镜头**，不带走人脸、
服装和画风。就是 LibTV「深度动作捕捉」那种输出。

<p align="center"><img src="docs/demo.gif" width="560" alt="左：原片；右：深度白模"></p>

一条命令，一个 Apache-2.0 的模型（111 MB），8G 显卡或 Apple M 系列芯片都能跑。

## 安装

```bash
pip install git+https://github.com/maosika-ai/video2blockout
# 需要 PATH 里有 ffmpeg（或 pip install "video2blockout[ffmpeg]"）
```

首次运行自动从 Hugging Face 下载权重。国内请设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 用法

```bash
video2blockout in.mp4 -o blockout.mp4                 # 保持原 fps / 尺寸，近白远黑
video2blockout in.mp4 -o blockout.mp4 --target h3     # MiniMax H3：24fps、边长 32 的倍数、≤15 秒
video2blockout in.mp4 -o blockout.mp4 --npz depth.npz # 同时保存原始浮点深度
```

```python
from video2blockout import extract, to_gray, write_gray_video
depths, fps = extract("in.mp4")                 # float32 [T, H, W]，越大越近
write_gray_video(to_gray(depths), "out.mp4", fps)
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--model small\|base\|large` | `small` | 只有 `small` 是 Apache-2.0；`base`/`large` 是 CC-BY-NC，加载时会警告 |
| `--target none\|h3\|wan` | `none` | 按目标模型设 fps 与尺寸（见下表） |
| `--fps N` | 原片 | 按**时间戳**选帧，30→24 真的是 24，不是整数步长那种假 24 |
| `--max-res N` | 1280 | 推理前把长边压到 N；输出也是这个尺寸 |
| `--invert` | 关 | 改成远白近黑 |
| `--clip PCT` | 0 | 两端各裁掉 PCT% 再归一，防一个热点像素把整段压暗 |
| `--gamma G` | 1.0 | >1 压暗中间调（近处层次更开） |
| `--crf N` | 12 | x264 质量。故意给高：控制视频里的色带会变成出片里的抖动 |
| `--npz PATH` | – | 保存原始深度，留给自己后处理 |

### 目标预设

| `--target` | fps | 尺寸 | 时长上限 | 对应 |
|---|---|---|---|---|
| `none` | 原片 | 偶数 | – | 通用 |
| `h3` | 24 | 32 的倍数（居中裁） | 15 秒 | [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) 深度输入 |
| `wan` | 16 | 16 的倍数 | – | Wan 2.1 VACE 控制视频 |

尺寸用裁不用补：补上去的黑边在生成模型眼里是一堵远墙。

## 做对的四个细节

1. **整段视频只归一化一次，不逐帧。** 逐帧 min/max 会让静止的墙在有人走近镜头时忽明忽暗，生成模型读到的就是"场景在呼吸"。
2. **时序一致性来自模型，不是靠模糊。** Video Depth Anything 用 32 帧窗口、10 帧重叠、跨窗口对齐尺度；我们原样调用它的推理路径，不在外面重新切段。
3. **近处白。** 深度 ControlNet 训练时用的就是这个约定；要反过来用 `--invert`。
4. **fps 按时间戳、尺寸靠裁剪、质量靠 CRF。** 见上表。

## 实测性能（全片端到端，Small 模型，294 帧 @ 736×1280）

| 设备 | 精度 | 速度 | 说明 |
|---|---|---|---|
| Apple M2 Max，torch 2.9.1 | fp32 | 约 500 ms/帧（12 秒片 2.7 分钟） | MPS 上 fp16 慢到不可用，已强制 fp32 |
| Apple M2 Max，torch 2.6.0 | fp32 | 约 1700 ms/帧 | 慢 3.5 倍——**Apple 芯片请升级 torch ≥ 2.9** |
| CPU（M2 Max） | fp32 | 约 1850 ms/帧 | 能用，慢 |
| NVIDIA RTX 4090（AutoDL），torch 2.8 cu128 | fp16 | **70 ms/帧**，`--input-size 364` 时 31 ms | 宿主内存峰值 4.5 GB（改成模型分辨率推理前是 8.2） |
| NVIDIA A100（官方数据） | fp16 | 约 8 ms/帧 | 32 帧批次占 6.8 GB 显存 |

## 模型与许可

| 权重 | 许可 | 商用 |
|---|---|---|
| `Video-Depth-Anything-Small`（默认） | Apache-2.0 | ✅ |
| `Video-Depth-Anything-Base` / `-Large` | CC-BY-NC-4.0 | ❌ 需显式指定，加载时警告 |

代码 Apache-2.0。模型代码来自
[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)，
vendored 在 `video2blockout/third_party/`（见 `NOTICE`）。

## ComfyUI

已有现成节点 [ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)。
本项目面向要 CLI / Python API、要许可证安全默认值、要目标模型预设的人。

## 路线图

- `--people-only`：只留人物剪影、背景抹平（SAM 2）
- `--also pose,canny,normal`：一遍出 MiniMax H3 Fun ControlNet 的其余几种输入
- Depth Anything 3（逐帧 + 平滑）与 ViGeo 后端
- Gradio 演示 / Hugging Face Space

## 致谢

Video Depth Anything — Chen et al., CVPR 2025。由 [猫斯卡](https://www.maosika.com) 出品。
