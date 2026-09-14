---
license: Apache License 2.0
language:
  - zh
  - en
domain:
  - cv
  - multi-modal
tasks:
  - video-depth-estimation
  - body-2d-keypoints
frameworks:
  - pytorch
  - onnx
tags:
  - reshot
  - maosika
  - 猫斯卡
  - seedance
  - minimax-h3
  - video-depth-anything
  - dwpose
  - openpose
  - controlnet
  - depth-map
  - ai-short-drama
  - AI短剧
  - 视频生成
  - 参考视频
---

<!--
  这是 ReShot 在魔搭（ModelScope）上的模型卡：modelscope.cn/models/maosika/reshot
  源文件在 GitHub 仓库 docs/MODELSCOPE.md，改这里再同步上传；图片走仓库里的相对路径。
  文案口径与 README.zh-CN.md 一致：讲意义和收获、Seedance 放首位、不提任何竞品。
-->

<h1 align="center">ReShot</h1>
<p align="center"><b>复制走位，不复制演员。</b></p>
<p align="center">ReShot 把一段参考视频变成深度图、骨架或线稿视频，让 Seedance 或 MiniMax H3 照着它的动作和运镜再拍一遍——人物换成你的。</p>

<p align="center"><img src="docs/demo-fight.gif" width="720" alt="一段武打参考片、它的深度图、用深度图生成的三段新武打片"></p>
<p align="center"><sub>上排：参考片和它的深度图。下排：用这张深度图生成的三段片——两个女人、一只兔子。动作一样，镜头一样。</sub></p>

<p align="center"><a href="https://github.com/maosika-ai/reshot">GitHub</a> · <a href="https://pypi.org/project/reshot/">PyPI</a> · <a href="https://github.com/maosika-ai/reshot/blob/main/docs/USAGE.zh-CN.md">使用手册</a> · <a href="https://github.com/maosika-ai/ComfyUI-ReShot">ComfyUI 节点</a> · <a href="https://www.maosika.com/blog/reshot-open-source-depth-motion-capture">为什么做它（猫斯卡官网）</a></p>

---

## 你遇到的问题

你看中了一段片子：那场打戏、那段舞、那个运镜，正是你的 AI 视频想要的。想拿过来用，有两条路，都走不通：

- **把片子直接当参考视频喂给模型。** 它会把脸、衣服、画风连同动作一起抄走。片子里如果是真人，平台审核可能直接拒收。
- **用文字描述动作。**「她蹬墙而起，扯下一根管子，把大个子过肩摔」——模型每次给你的打法都不一样，镜头更是从来不听话。

## ReShot 做什么

ReShot 吃进一个 `.mp4`，吐出一个 `.mp4`。输出是控制视频，三种任选：

| `--control` | 出来的是什么 | 留下 | 扔掉 | 什么时候选它 |
|---|---|---|---|---|
| **`depth`**（默认） | 灰片，近白远黑 | 谁站在哪、谁大谁小、每个动作、运镜、场景的形状 | 脸、衣服、光线、画风 | 什么都行：打戏、走位、运镜、群戏 |
| **`pose`** | OpenPose 画法的骨架——黑底彩色火柴人，带手、不画脸 | 每一根肢体、每一只手，分毫不差 | 其余全部，包括体型和场景 | 跳舞、武打，凡是靠肢体的；MiniMax H3 Fun ControlNet 的 `pose` 口 |
| **`canny`** | 黑底白线的边缘图 | 构图和轮廓 | 颜色和明暗——但脸型和服装的轮廓**留着** | 想要整个画面的布局、不只是人的时候 |

<p align="center"><img src="docs/img/step1_reference.jpg" width="300" alt="参考片的一帧"> <img src="docs/img/step2_depth.jpg" width="300" alt="同一帧的深度图"> <img src="docs/img/step2_pose.jpg" width="300" alt="同一帧的骨架"></p>
<p align="center"><img src="docs/demo-pose.gif" width="720" alt="武打参考片、它的骨架、用骨架生成的三段新片"></p>
<p align="center"><sub>同一个演示，参考换成<b>骨架</b>。上排：参考片和它的骨架视频。下排：MiniMax H3 用这段骨架出的三段片——提示词和定妆图与深度版完全一样，只换了 <code>&lt;Video 1&gt;</code>。</sub></p>

你把这段视频当参考交给视频模型，提示词里写人物和画风。动作和镜头模型从片里读，其余全听你的。

严格地说：深度图是单目视频深度估计（Video Depth Anything），对整段视频做一次归一化变成 8 位灰度；骨架是 DWPose（YOLOX 检人 + RTMPose 全身估点），跨帧做身份跟踪、可见性迟滞和 One-Euro 平滑，再按视频 ControlNet 训练时见过的 OpenPose 画法画出来；线稿就是 OpenCV，不用模型。

## 怎么用

### 1. 装

```bash
pip install "reshot[pose]"        # 需要 PATH 里有 ffmpeg；没有就装 "reshot[pose,ffmpeg]"
```

模型权重首次运行自动下载（深度 111 MB，骨架 340 MB）。国内先设一个镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

也可以把权重放本地，`--checkpoint 深度权重.pth`、`--checkpoint-dir 骨架权重目录/` 直接指过去，不走网络。

用 Claude Code、Codex、Cursor 之类的 AI 编程工具？把下面这段粘给它，它会替你装好、验显卡、跑一段测试片：

```
在这台机器上安装 ReShot（PyPI 上的 "reshot" 包）并跑通。
先读 https://raw.githubusercontent.com/maosika-ai/reshot/main/docs/AGENT_INSTALL.md，按它一步步做：
识别系统和显卡，先装对应的 PyTorch，再 pip install "reshot[ffmpeg]"；
我在中国大陆，先设 HF_ENDPOINT=https://hf-mirror.com；跑一次假后端冒烟测试，再用一段短片真跑一次并加 --metrics，把数字给我看。
depth.mp4 没生成出来之前不要说完成。
```

### 2. 出控制视频

**最省事——网页。** 只敲 `reshot`，后面什么都不加，浏览器会打开一个页面（所有处理都在你自己电脑上）。把片子拖进去，选**出什么**（深度图 / 骨架 / 线稿）和给哪家模型用，点按钮。参考片和结果并排预览，数字、下载按钮、要粘到 Seedance 或 MiniMax H3 的提示词都在同一屏。

<p align="center"><img src="docs/img/web_ui_zh.jpg" width="720" alt="ReShot 网页：左边参考片与骨架并排，右边选出什么、给谁用和画质"></p>

**或者命令行**，适合脚本和批量：

```bash
reshot 参考片.mp4 -o 深度图.mp4 --target seedance                  # 深度图（默认）
reshot 参考片.mp4 -o 骨架.mp4  --target h3 --control pose         # 骨架
reshot 参考片.mp4 -o out/      --control depth,pose,canny         # 三种一起出
reshot clips/*.mp4 -o depth/ --target seedance                    # 整个文件夹，模型只加载一次
```

`--target seedance` 会把输出设成 24 fps、H.264、边长 16 的倍数、不少于 407,696 像素、最长 15 秒——这就是 Seedance 接口对参考视频的要求。给 MiniMax H3 用 `--target h3`。

### 3. 交给视频模型

**Seedance 2.0 / 2.5。** 把 `深度图.mp4` 当参考视频上传，提示词里点名它，再写人物和画风：

```
参考@视频1的动作与运镜，顺序与视频保持一致。
一名穿深绿色丝绒旗袍的女子在狭窄的金属走廊里与三名黑衣守卫搏斗，冷蓝走廊光，红色警示灯，电影感。
```

挂的是骨架视频就明说，并且说清线条不要出现：`参考@视频1里火柴人的动作与运镜，顺序与视频保持一致；只取动作和机位，画面里不要出现骨架线条。`

**MiniMax H3。** 把 `深度图.mp4` 挂成 `<Video 1>`，定妆图挂成 `<Picture 1>`。关键是两句：用文字把灰片里发生的事写一遍，再明说灰色外观不要抄。演示用的三张定妆图和完整提示词（带 seed）都在 GitHub 仓库的 [`docs/prompts/`](https://github.com/maosika-ai/reshot/tree/main/docs/prompts)。

<p align="center"><img src="docs/img/step3_sheets.jpg" width="720" alt="演示用的三张定妆图"></p>

**ComfyUI。** 装 [ComfyUI-ReShot](https://github.com/maosika-ai/ComfyUI-ReShot)，在 Load Video 和你的模型之间放一个 *ReShot Depth Video* 节点就行，完全不用命令行。

### 4. 出来的是什么

<p align="center"><img src="docs/img/step4_takes.jpg" width="720" alt="三段生成片"></p>

从左到右：穿青铜甲的江雪、穿墨绿丝绒旗袍的苏晚、和狼、虎、熊打成一团的拳手兔子（院线 3D 动画风）。同样的六个镜头，开头同样的特写，结尾同样走出那扇门。参考片本身是 Seedance 2.0 用文字生成的（864×496，12 秒），三段片在 MiniMax H3 上出的，全程没有任何真人肖像。

## 能复刻什么，不能复刻什么

**能：** 谁站在哪、彼此谁大谁小、每个动作和它的节奏、切镜、运镜——推进、跟拍、手持抖动。

**不能：** 脸（用定妆图）、衣服、光线、颜色、道具细节，以及比手还小的东西。这些全靠你的提示词和参考图。

**深度图还是骨架？** 深度图带走整个画面——人和非人都管用。骨架只带人，但每根肢体每根手指都准。跳舞、武打选骨架，其余选深度图；拿不准就两个都出（`--control depth,pose`）。大特写别用骨架：肩和髋不在画面里时估计器照样会猜——特写老老实实用深度图。

做演示时踩出来的几条：

- **参考片控制在 15 秒以内**（Seedance 的上限），跑 ReShot 之前先把片子剪到你要的那几个镜头。
- **给 MiniMax H3 的深度图要缩小：`--target h3 --max-res 320`**。全尺寸的灰色人影会把角色的脸型往参考片里那个人上带。
- **换物种可以，体量比例别动。** 谁大谁小深度图已经定了，提示词不能跟它打架。
- **配角穿素色衣服，不留标志。** 提示词没写死的地方，模型会自己往上填字和徽章。

## 什么机器能跑

| | 能不能跑 | 实测 |
|---|---|---|
| **N 卡 8 GB 及以上** | 能 | 默认 `--quality fast`：显存 3 GB、3080 Ti 上 34 ms/帧 |
| **N 卡 12 GB 及以上** | 能，`--quality full` 也放得下 | full：显存 11 GB、4090 上 62 ms/帧 |
| **Apple 芯片** | 能 | M2 Max 上 46 ms/帧，12 秒的片约 18 秒 |
| **纯 CPU** | 能，很慢 | 约 1.8 秒/帧 |
| **`--control pose`** | 以上都行 | M2 Max 走 CPU：306 ms/帧，12 秒的片 90 秒 |
| **`--control canny`** | 什么都行 | 不用模型；12 秒的片不到一秒 |

## 预设

| `--target` | fps | 尺寸 | 时长 | 给谁用 |
|---|---|---|---|---|
| `seedance` | 24 | 16 的倍数，≥ 407,696 像素 | ≤ 15 秒 | Seedance 2.0 / 2.5 参考视频 |
| `h3` | 24 | 32 的倍数 | ≤ 15 秒 | MiniMax H3（参考视频或 Fun ControlNet 深度条件） |
| `wan` | 16 | 16 的倍数 | – | Wan 2.1 VACE |
| `none` | 原片 | 偶数 | – | 任何认深度视频的模型 |

## 给开发者

```python
from pathlib import Path
from reshot import RunConfig, run

run(RunConfig(input=Path("参考片.mp4"), output=Path("深度图.mp4"), target="seedance"))
```

四个细节决定了视频模型愿不愿意跟着这段片走：**整段只用一把尺子**（深度对全部帧归一化一次，绝不逐帧）、**骨架不闪**（跨帧跟踪 + 迟滞 + One-Euro）、**按时间戳选帧**、**只裁不补**（黑边会被当成远处的墙）。

## 模型与协议

| 用途 | 模型 | 协议 |
|---|---|---|
| 深度图 | Video Depth Anything **Small**（字节跳动，CVPR 2025，28M 参数） | Apache-2.0 |
| 骨架 | DWPose（YOLOX-L 检人 + RTMPose 全身估点，ONNX） | Apache-2.0 |
| 线稿 | OpenCV Canny | 不用模型 |

ReShot 本身 Apache-2.0，默认深度模型和骨架模型（代码和权重）也都是。用在产品、流水线、服务里都行。更大的研究用深度权重（Base、Large）是 CC-BY-NC，不主动指定不会加载。

代码、issue、讨论区都在 **[github.com/maosika-ai/reshot](https://github.com/maosika-ai/reshot)**。做出片了，欢迎到[作品分享](https://github.com/maosika-ai/reshot/discussions/2)贴参考片、深度图、成片和你的提示词那一行。

## 关于猫斯卡

ReShot 由 **[猫斯卡](https://www.maosika.com)**（www.maosika.com）开源。猫斯卡是一套专业的
**AI 视频自动生产系统**，专注 **AI 短剧、AI 短视频、AI 漫剧**的全流程制作：从一句话创意出发，
自动写出分集剧本，设计人物与场景，生成前后一致的角色定妆图和场景图，再用 **Seedance 2.0 / 2.5**、
**MiniMax H3** 等视频模型把每一镜拍出来——每个环节都由对应的数字专家接手，一个人就能完成过去
一个团队才能做的竖屏短剧。个人编剧、MCN 机构和短剧公司每天都在用猫斯卡生产 AI 短剧。

ReShot 是这条生产线里的「深度图」环节，以 Apache-2.0 协议开源，任何人都可以用它把参考镜头的
走位和运镜复制到自己的 AI 视频里。为什么做它、三段样片和提示词的完整讲解：
**[猫斯卡官网上的 ReShot 文章](https://www.maosika.com/blog/reshot-open-source-depth-motion-capture)**（[English](https://www.maosika.com/blog/reshot-open-source-depth-motion-capture-seedance-minimax-h3)）。
想端到端做 AI 短剧、AI 短视频、AI 漫剧，请访问 **[https://www.maosika.com](https://www.maosika.com)**。

<p align="center"><sub>出品：<a href="https://www.maosika.com">猫斯卡</a> · 每天都在出 AI 短剧 · <a href="https://www.maosika.com">www.maosika.com</a></sub></p>
