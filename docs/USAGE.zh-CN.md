# ReShot 使用手册

[English](USAGE.md)

1. [安装](#安装)
2. [命令行](#命令行)
3. [Python 接口](#python-接口)
4. [实战配方](#实战配方)
5. [原理](#原理)
6. [排错](#排错)
7. [常见问题](#常见问题)

## 安装

要求：Python ≥ 3.10、PyTorch ≥ 2.1（CPU / CUDA / Apple MPS 皆可）、ffmpeg。

```bash
# 自己有 ffmpeg
pip install git+https://github.com/maosika-ai/reshot

# 没有 ffmpeg？这个 extra 自带一个静态二进制
pip install "reshot[ffmpeg] @ git+https://github.com/maosika-ai/reshot"
```

PyTorch 不锁 CUDA 版本；如果 `pip` 装错了，先按 https://pytorch.org/get-started/locally/ 装好再装 reshot。

模型权重（111 MB）首次运行时从 Hugging Face 下到 `~/.cache/huggingface`。
离线机器：把这个缓存目录拷过去，或用 `--checkpoint /路径/video_depth_anything_vits.pth`。
国内：`export HF_ENDPOINT=https://hf-mirror.com`。

## 命令行

```
reshot 输入 -o 输出 [选项]
```

### 模型

| 选项 | 默认 | 说明 |
|---|---|---|
| `--model small\|base\|large` | `small` | 只有 `small` 是 Apache-2.0。`base`/`large` 是 CC-BY-NC-4.0：研究可用，运行时会警告。 |
| `--device auto\|cuda\|mps\|cpu` | `auto` | CUDA 用 fp16，其余 fp32（MPS 上 fp16 慢到不可用）。 |
| `--quality Q` | `fast` | 模型工作的分辨率。`fast` = 16:9 是 644×364、9:16 是 364×644（约 3 GB 显存）；`full` = 924×518 / 518×924（约 11 GB、慢 2.5 倍、细轮廓更锐：平均差 5.3/255、边缘 −4.5%）。不影响输出尺寸。 |
| `--input-size N` | — | 专家用：精确指定模型短边，14 的倍数，覆盖 `--quality`。 |
| `--checkpoint 路径` | – | 本地 `.pth`，跳过下载。 |

### 输出

| 选项 | 默认 | 说明 |
|---|---|---|
| `--target none\|seedance\|h3\|wan` | `none` | fps + 尺寸预设，见[目标预设](#目标预设)。 |
| `--fps N` | 预设或原片 | 按**时间戳**选帧，30→24 真的是 24。绝不向上插帧。 |
| `--max-res N` | `1280` | 输出长边上限。推理永远在模型分辨率进行，与此无关。 |
| `--max-frames N` | 全部 | 只处理前 N 帧。 |
| `--invert` | 关 | 改成远白近黑。默认近白远黑（深度 ControlNet 要的就是这个）。 |
| `--clip PCT` | `0` | 两端各裁 PCT% 再拉到 0–255，防一个热点像素压暗全片。`0` 为精确 min/max。 |
| `--gamma G` | `1.0` | >1 压暗中间调，近处层次更开。 |
| `--crf N` | `12` | x264 质量。故意给高：控制视频里的色带会变成出片里的抖动。 |
| `--npz 路径` | – | 处理分辨率下的原始 float32 深度（`depths[T,H,W]`、`fps`）。 |
| `--metrics 路径` | – | JSON：配置、计划、耗时、峰值内存、输出尺寸。 |

### 安全

| 选项 | 说明 |
|---|---|
| `--force` | 内存估算超过物理内存一半时仍然运行。 |

### 退出码

| 码 | 含义 | 怎么办 |
|---|---|---|
| 0 | 完成 | |
| 1 | 意外错误（是 bug——请带 `-v` 输出提 issue） | |
| 2 | 输入或参数有问题 | 按报错里指出的路径 / 参数改 |
| 3 | 内存估算超预算 | 用建议的 `--max-frames`、把片子切短，或 `--force` |
| 4 | 缺 ffmpeg | `pip install imageio-ffmpeg` 或装 ffmpeg |
| 5 | 拿不到权重 | `HF_ENDPOINT` 指向镜像，或 `--checkpoint` |

### 目标预设

| `--target` | fps | 尺寸 | 时长上限 | 说明 |
|---|---|---|---|---|
| `none` | 原片 | 偶数 | – | |
| `seedance` | 24 | 16 的倍数（居中裁），≥ 407,696 像素（不够警告） | 15 秒（超了警告） | Seedance 2.0 / 2.5 参考视频（`@视频N` 继承动作与运镜） |
| `h3` | 24 | 32 的倍数（居中裁） | 15 秒（超了警告） | MiniMax-H3-Fun-Controlnet-Union 深度输入 |
| `wan` | 16 | 16 的倍数 | – | Wan 2.1 VACE 控制视频；按文档取值，尚未端到端实测 |

尺寸用**裁**不用补：补上去的黑边在生成模型眼里是一堵远墙。

## Python 接口

```python
from pathlib import Path
from reshot import RunConfig, run, run_many, plan

cfg = RunConfig(input=Path("in.mp4"), output=Path("out.mp4"), target="h3", metrics=Path("m.json"))
print(plan(cfg))          # 尺寸、帧数、内存估算——此时还没解码
result = run(cfg)         # 与命令行完全同一条代码路径
print(result.frames, result.fps, result.ms_per_frame, result.peak_rss_bytes)
```

更底层的零件，给自己做后处理用：

```python
from reshot import extract, to_gray, upsample_frames, write_gray_video

depths, fps = extract("in.mp4")                     # float32 [T, H, W]，越大越近
gray = to_gray(depths, clip_percent=0.5, gamma=1.2)  # uint8 [T, H, W]，整段归一
write_gray_video(upsample_frames(gray, 1280, 736), "out.mp4", fps, size=(1280, 736))
```

属于"用户该修"的错误都是 `reshot.ReshotError` 的子类、带 `exit_code`；其余都是 bug。

`RunConfig(..., backend="fake")` 用合成深度场跑完整条流水线、不载模型——几毫秒验证你自己的集成。

## 实战配方

### Seedance 2.0 / 2.5（参考视频）

1. `reshot 参考.mp4 -o 参考_depth.mp4 --target seedance`（24 fps、16 倍数、≤15 秒）。
2. 作为**参考视频**提交（API 里 `role: "reference_video"`；控制台里当视频参考挂上），
   提示词用 `@视频1` 指向它，要它的动作与运镜：

   ```
   参考@视频1的动作与运镜。两名武者在雨夜屋顶对决，黑色劲装，冷蓝月光，电影感。
   ```

   提示词只写**长相**——人物、服装、光线、画风。走位和镜头由视频决定。
3. 官方对参考视频的要求，以及预设替你做了什么：

   | 官方要求 | 预设 |
   |---|---|
   | mp4 / mov，H.264 或 H.265 | H.264 mp4，yuv420p |
   | 24 – 60 fps | 24 fps |
   | 2.0：单段 2 – 15 秒、总计 ≤ 15 秒 · 2.5：单段 2 – 30 秒、总计 ≤ 30 秒 | 超 15 秒警告 |
   | 宽×高 ≥ 407,696 像素，单边 300 – 6,000，宽高比 0.4 – 2.5 | 低于像素下限警告——调大 `--max-res` 或换更大的原片 |
   | ≤ 200 MB | CRF 12 下 15 秒 1080p 远小于此 |

   Seedance 2.5 允许到 30 秒：需要更长时不加 `--target`、只传 `--fps 24`，尺寸自己核一下。

### MiniMax H3 Fun ControlNet（ComfyUI）

1. `reshot 参考.mp4 -o 参考_depth.mp4 --target h3`（24 fps、32 倍数、≤15 秒）。
2. ComfyUI 里装 [Fun ControlNet Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) 权重，
   把 `参考_depth.mp4` 作为控制视频、条件选 **depth**，提示词只写**长相**（人物、服装、光线、画风）。
   走位和镜头由控制视频决定。
3. 生成分辨率和控制视频保持一致，或让节点缩放——但别补边。

### Wan 2.1 VACE

`reshot 参考.mp4 -o 参考_depth.mp4 --target wan`，在 VACE 工作流里当深度控制视频用。
这个预设还没端到端验过，fps/尺寸如需调整欢迎反馈。

### 批量

一次给 `reshot` 多个输入和一个目录，模型只加载一次，每条片输出为 `<原名>_depth.mp4`。
`--metrics` / `--npz` 此时也当目录用（`<原名>.json` / `<原名>.npz`）。某条片失败会报出来并跳过，
退出码取第一个失败的。

```bash
reshot clips/*.mp4 -o depth/ --target seedance --metrics depth/metrics/
```

Python 里对应 `run_many()`：

```python
from pathlib import Path
from reshot import RunConfig, run_many

cfgs = [RunConfig(input=p, output=Path("depth") / f"{p.stem}_depth.mp4", target="seedance") for p in Path("clips").glob("*.mp4")]
for r in run_many(cfgs):        # 成功是 RunResult，失败是那条片的 ReshotError
    print(r)
```

### 长视频

整段视频的帧和深度都在内存里。reshot 会先估算，超过物理内存一半就拒跑并给 `--max-frames` 建议。
三分钟的片子先用 `ffmpeg -ss … -t 15 …` 切成 ≤15 秒的段——生成模型本来也只吃短的控制视频。

### Apple 芯片

开箱即用（`--device auto` 会选 MPS）。M2 Max 上两条实测：torch 2.9.1 比 2.6.0 快约 3.5 倍；
MPS 上 fp16 跑不完，所以强制 fp32。736×1280 约 0.5 秒/帧，12 秒的片约 2.5 分钟。

### 可复现的深度数据（研究用）

`--npz` 输出处理分辨率下的原始浮点深度；配合 `--metrics`（记录完整配置与计划），一次运行可完全复现。

## 原理

```
输入视频 ─▶ 探测 ─▶ 计划（尺寸、帧数、内存）─▶ 按模型分辨率解码
        ─▶ Video Depth Anything（32 帧窗口、10 帧重叠、跨窗口对齐）
        ─▶ 整段归一化 → uint8，近白远黑
        ─▶ 逐帧放大到输出尺寸 ─▶ 居中裁到目标倍数 ─▶ x264
```

- **模型**：[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)（CVPR 2025）。
  它的时序模块是深度不闪的原因；单图模型逐帧跑必闪。模型代码原样 vendored 在 `third_party/`（只改两行 import）。
- **分辨率**：模型工作在 `--quality` 定的尺寸——16:9 的片 fast 是 644×364、full 是 924×518。喂更大的帧只会让每帧存的浮点深度变大；reshot 按这个尺寸解码，最后把 8 位结果放大回原片尺寸。
- **归一化**：`(d − min) / (max − min)` 对**全部帧**做一次；可选百分位裁剪。
- **编码**：`libx264 -crf 12 -pix_fmt yuv420p -g 2·fps -movflags +faststart`。

## 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: torchvision` | torch 装了但没装 torchvision | `pip install torchvision`（与 torch 同一构建） |
| 退出码 4 / "ffmpeg not found" | 没 ffmpeg | `pip install imageio-ffmpeg` 或装 ffmpeg |
| 退出码 5 / 权重下载失败 | 连不上 huggingface.co | `export HF_ENDPOINT=https://hf-mirror.com`，或 `--checkpoint` |
| 退出码 3 | 片子太长内存不够 | 用建议的 `--max-frames`，或切片 |
| Mac 上极慢 | torch < 2.9，或强开了 fp16 | `pip install -U torch`；MPS 上别开 fp16 |
| 给了 `--fps 24` 输出还是 30 | 原片 fps 低于 24 | reshot 不向上插帧 |
| 输出比原片小 | `--target` 裁剪或 `--max-res` | 正常，见目标预设 |
| 深度图有色带 | 调高了 `--crf` | 保持 ≤14 |
| 人和背景糊在一起 | 场景纵深本身很小 | 试 `--clip 0.5 --gamma 1.3` |

## 常见问题

**Small 够用吗？** 做控制视频够：生成模型读的是轮廓、体量和运动，不是精细几何，Small 的剪影已经很干净（见首页动图）。
Base/Large 和 Small 的差距我们没有自己测过；它们是 CC-BY-NC，商用本来也只能选 Small。

**为什么近白远黑？** 深度 ControlNet（H3 Fun、Wan VACE、SD depth）训练用的都是"越近越亮"的逆深度图。
要反过来用 `--invert`。

**能出骨架 / 边缘 / 法线吗？** 还不能，在路线图上。

**能去背景、只留人吗？** 还不能（`--people-only` 在路线图上）。全画面深度本身已经不含任何身份和画风信息。

**显存？** Small fp16 默认输入尺寸约 2 GB；4090 大部分时间在等 CPU。
