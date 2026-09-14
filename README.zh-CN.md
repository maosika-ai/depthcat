<h1 align="center">ReShot</h1>
<p align="center"><b>复制走位，不复制演员。</b></p>
<p align="center">ReShot 把一段参考视频变成深度图、骨架或线稿视频，让 Seedance 或 MiniMax H3 照着它的动作和运镜再拍一遍——人物换成你的。</p>

<p align="center"><img src="docs/demo-fight.gif" width="720" alt="一段武打参考片、它的深度图、用深度图生成的三段新武打片"></p>
<p align="center"><sub>上排：参考片和它的深度图。下排：用这张深度图生成的三段片——两个女人、一只兔子。动作一样，镜头一样。<a href="docs/demo-fight.mp4">高清原片</a>。</sub></p>

<p align="center"><a href="README.md">English</a> · <a href="docs/USAGE.zh-CN.md">使用手册</a> · <a href="https://github.com/maosika-ai/ComfyUI-ReShot">ComfyUI 节点</a> · <a href="https://huggingface.co/spaces/maosika/reshot">Hugging Face</a> · <a href="https://github.com/maosika-ai/reshot/discussions">社区</a> · <a href="CHANGELOG.md">更新日志</a></p>
<p align="center">
<a href="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/%E8%BF%90%E8%A1%8C%E4%BA%8E-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>

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
<p align="center"><sub>同一个演示，参考换成<b>骨架</b>。上排：参考片和它的骨架视频。下排：MiniMax H3 用这段骨架出的三段片——提示词和定妆图与深度版完全一样，只换了 <code>&lt;Video 1&gt;</code>。<a href="docs/demo-pose.mp4">高清原片</a> · 带 seed 的提示词在 <a href="docs/prompts/">docs/prompts/</a>（<code>*_pose.txt</code>）。参考 · 深度 · 骨架四张静帧：<a href="docs/img/controls_strip.jpg">controls_strip.jpg</a>。</sub></p>

你把这段视频当参考交给视频模型，提示词里写人物和画风。动作和镜头模型从片里读，其余全听你的。

严格地说：深度图是单目视频深度估计（Video Depth Anything），对整段视频做一次归一化变成 8 位灰度；骨架是 DWPose（YOLOX 检人 + RTMPose 全身估点），跨帧做身份跟踪、可见性迟滞和 One-Euro 平滑，再按视频 ControlNet 训练时见过的 OpenPose 画法画出来；线稿就是 OpenCV，不用模型。

## 怎么用

演示里那三段片就是这么做出来的，用到的每一个文件都在这个仓库里，你可以照着复现。

### 1. 装

**不想装：**[在 Colab 里打开笔记本](https://colab.research.google.com/github/maosika-ai/reshot/blob/main/examples/reshot_colab.ipynb)，免费 GPU，传一段视频、下载 depth.mp4。<a href="https://colab.research.google.com/github/maosika-ai/reshot/blob/main/examples/reshot_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

```bash
pip install reshot        # 需要 PATH 里有 ffmpeg
```

没有 ffmpeg 就装 `pip install "reshot[ffmpeg]"`，自带一个。连 Python 都没配？用 [uv](https://docs.astral.sh/uv/) 一行搞定：`uvx --from "reshot[ffmpeg]" reshot 参考片.mp4 -o 深度图.mp4 --target seedance`。模型权重 111 MB，首次运行自动下载；国内先 `export HF_ENDPOINT=https://hf-mirror.com`。

### 或者让你的 AI 编程工具来装

用 Claude Code、Codex、Cursor 之类的 AI 工具？把下面这段粘给它，它会替你装好、验显卡、跑一段测试片：

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

```bash
reshot                     # 打开 http://127.0.0.1:8765，结果存到 ~/ReShot
reshot web --port 9000 --out ./depth --no-browser     # 想改端口 / 目录 / 不自动开浏览器
```

**或者命令行**，适合脚本和批量：

```bash
reshot 参考片.mp4 -o 深度图.mp4 --target seedance                  # 深度图（默认）
reshot 参考片.mp4 -o 骨架.mp4  --target h3 --control pose         # 骨架；先 pip install "reshot[pose]"
reshot 参考片.mp4 -o out/      --control depth,pose,canny         # 三种一起出，out/参考片_<control>.mp4
```

`--target seedance` 会把输出设成 24 fps、H.264、边长 16 的倍数、不少于 407,696 像素、最长 15 秒——这就是 Seedance 接口对参考视频的要求。给 MiniMax H3 用 `--target h3`（边长 32 的倍数）。RTX 4090 上 12 秒的片深度图约 20 秒跑完；MacBook 深度图要几分钟、骨架约一分半（骨架模型在 Mac 上走 CPU）。整个文件夹一起来、模型只加载一次：`reshot clips/*.mp4 -o depth/ --target seedance`。

`--control pose` 加 `--keypoints 骨架.json` 会把骨架数据也存成 JSON（每帧每个人 134 个点带分数、每个人一个稳定的 id），可以拿去改、重定向、做分析；网页上同一份文件有下载按钮。

### 3. 交给视频模型

**Seedance 2.0 / 2.5。** 把 `深度图.mp4` 当参考视频上传，提示词里点名它，再写人物和画风：

```
参考@视频1的动作与运镜，顺序与视频保持一致。
一名穿深绿色丝绒旗袍的女子在狭窄的金属走廊里与三名黑衣守卫搏斗，冷蓝走廊光，红色警示灯，电影感。
```

挂的是骨架视频就明说，并且说清线条不要出现：`参考@视频1里火柴人的动作与运镜，顺序与视频保持一致；只取动作和机位，画面里不要出现骨架线条。`

**ComfyUI。** 装 [ComfyUI-ReShot](https://github.com/maosika-ai/ComfyUI-ReShot)，在 Load Video 和你的模型之间放一个 *ReShot Depth Video* 节点就行，完全不用命令行。

**MiniMax H3。** 把 `深度图.mp4` 挂成 `<Video 1>`。想要固定的脸，再把定妆图挂成 `<Picture 1>`。演示用的三张定妆图：

<p align="center"><img src="docs/img/step3_sheets.jpg" width="720" alt="演示用的三张定妆图"></p>

MiniMax H3 的提示词有固定的六段格式，三段片的完整提示词都在 [`docs/prompts/`](docs/prompts/)。真正起作用的是这两句——怎么定义 `<Video 1>`，允许它转移什么：

```
<Subject 3> is the fight choreography and camera movement shown in <Video 1>, a grey depth map
in which near objects are white and far objects are black: one fighter leans on a corridor wall
in close-up, kicks off it to tear down a pipe, fights several opponents, is grabbed from behind
by the largest and throws him, slams the last one into a wall panel, wipes the mouth in close-up,
then walks away through a door past the fallen opponents.

<Subject 3>: attribute_transfer - every action, position, timing and camera move of <Video 1>
is transferred onto <Subject 1> and <Subject 2>; its grey depth look is not transferred.
```

两个要点。**用文字把灰片里发生的事写一遍**——提示词告诉它那些灰影在干什么，模型读深度图会准得多。**明说灰色外观不要抄**，不然可能给你出一部灰片。

换成骨架视频，这两句就写成 `<Video 1>, an OpenPose skeleton video — coloured stick figures on black, one per person: …` 和 `… the stick-figure look is not transferred.`。在 ComfyUI 里配 Fun ControlNet 权重时更省事：`骨架.mp4` 直接接 ControlNet 的 **pose** 口，不用在提示词里绕。网页会按你出的是哪种，打印对应的那几句。

### 4. 出来的是什么

<p align="center"><img src="docs/img/step4_takes.jpg" width="720" alt="三段生成片"></p>

从左到右：穿青铜甲的[江雪](docs/prompts/take1_jiangxue_armor.txt)、穿墨绿丝绒旗袍的[苏晚](docs/prompts/take2_suwan_qipao.txt)、和狼、虎、熊打成一团的[拳手兔子](docs/prompts/take3_rabbit_boxer.txt)（院线 3D 动画风）。同样的六个镜头，开头同样的特写，结尾同样走出那扇门。参考片本身是 Seedance 2.0 用文字生成的（864×496，12 秒），三段片在 MiniMax H3 上出的，全程没有任何真人肖像。

## 能复刻什么，不能复刻什么

**能：** 谁站在哪、彼此谁大谁小、每个动作和它的节奏、切镜、运镜——推进、跟拍、手持抖动。

**不能：** 脸（用定妆图）、衣服、光线、颜色、道具细节，以及比手还小的东西。这些全靠你的提示词和参考图。

**深度图还是骨架？** 深度图带走整个画面——墙在哪、谁更大、一根管子被扯下来——人和非人都管用。骨架只带人，但每根肢体每根手指都准，体型和场景一点不带。跳舞、武打选骨架，其余选深度图；拿不准就两个都出（`--control depth,pose`），看模型跟哪个跟得更好。我们在 MiniMax H3 上把演示的三段片用骨架当 `<Video 1>` 重出了一遍（提示词、定妆图完全一样）：三条都复现了六个节拍，和深度版一个水平，过肩摔那一镜骨架版更清楚——见上面第二张动图。每种条件只出了一条、seed 不同，所以只能读成「骨架当参考视频也行」，不是「骨架更好」。

我们在 MiniMax H3 上用演示的三段片做了实测：提示词、定妆图完全一样，只把 `<Video 1>` 从深度图换成骨架视频（连带改了描述它的两句话）。三条骨架版都复现了六个节拍——特写、群殴、过肩摔、撞控制台、跨过倒地守卫走出门——和深度版一个水平，过肩摔那一镜骨架版更清楚。每种条件只出了一条、seed 不同，所以只能读成「骨架当参考视频也行」，不能读成「骨架更好」。带 seed 的提示词在 [`docs/prompts/`](docs/prompts/)（`*_pose.txt`）。

<p align="center"><img src="docs/img/pose_vs_depth_takes.jpg" width="864" alt="第一行参考片；之后每个角色两行：深度版成片和骨架版成片，各六镜"></p>
<p align="center"><sub>最上面是参考片。往下每个角色两行：用深度图出的片、用骨架出的片，各六个镜头。</sub></p>有一种情况骨架不合适：大特写。肩和髋不在画面里时估计器照样会猜，把它们画在画面边上——特写老老实实用深度图。线稿会把服装和脸的轮廓也带过去，「只复制走位」对它只成立一半。

做演示时踩出来的几条：

- **参考片控制在 15 秒以内**（Seedance 的上限），跑 ReShot 之前先把片子剪到你要的那几个镜头。片子里有什么就会复刻什么，包括结尾没用的那段。
- **给 MiniMax H3 的深度图要缩小：`--target h3 --max-res 320`**（16:9 就是 320×176）。全尺寸的灰色人影会把角色的脸型往参考片里那个人上带；缩小后只带动作、不带脸型。
- **换物种可以，体量比例别动。** 熊那一段能成，是因为提示词写了熊「约为兔子的 1.3 倍，不超过 1.5 倍」。谁大谁小深度图已经定了，提示词不能跟它打架。
- **配角穿素色衣服，不留标志。** 提示词没写死的地方，模型会自己往上填字和徽章。

## 什么机器能跑

| | 能不能跑 | 实测 |
|---|---|---|
| **N 卡 8 GB 及以上**（RTX 3070、4060、4090……） | 能 | 默认 `--quality fast`：显存 3 GB、3080 Ti 上 34 ms/帧 |
| **N 卡 12 GB 及以上** | 能，`--quality full` 也放得下 | full：显存 11 GB、3080 Ti 上 83 ms/帧、4090 上 62 ms/帧 |
| **Apple 芯片** | 能 | 默认 `fast`：M2 Max 上 46 ms/帧，12 秒的片约 18 秒 |
| **纯 CPU** | 能，很慢 | 约 1.8 秒/帧 |
| **内存** | 16 GB 能跑 720p 约 27 秒 | 峰值 = 2 GB + 每秒 720p 约 224 MB；放不下会在开始之前拒绝 |
| **`--control pose`** | 以上都行；要先 `pip install "reshot[pose]"`（ONNX Runtime；N 卡装 `onnxruntime-gpu`） | M2 Max 走 CPU：306 ms/帧，12 秒的片 90 秒；两个权重文件共 340 MB，只下一次。内存约 1.4 GB |
| **`--control canny`** | 什么都行 | 不用模型；12 秒的片不到一秒 |

**`--quality`——模型工作的分辨率。** 输出视频永远和原片一样大；这个选项定的是深度模型看到的那张图有多大。

| `--quality` | 模型看到的画面（16:9 · 9:16 · 4:3） | 显存 | 速度 | 细节 |
|---|---|---|---|---|
| `fast`（默认） | 644×364 · 364×644 · 490×364 | 约 3 GB | 3080 Ti 上 34 ms/帧 | 大形状与 full 一模一样；脸颊旁一缕散发会糊进脸里 |
| `full` | 924×518 · 518×924 · 686×518 | 约 11 GB | 3080 Ti 上 83 ms/帧 | 细轮廓更锐 |

网页上「画质」就是这两档；命令行用 `--quality`。画质只管深度图——骨架模型的工作尺寸是固定的（检人 640 px，估点每人 288×384），线稿按输出尺寸逐像素算。

**骨架的速度旋钮——`--pose-detect-every N`（默认 3）。** CPU 上检人占骨架总耗时的 80%，所以检测器每三帧跑一次，中间两帧的人框由上一帧的骨架推出来；片子里出现剪切、或某个骨架分数掉到不可信，检测器立刻回来。在 289 帧的走廊片上和逐帧检测对比：280 帧人数完全一致，80% 的关节误差在 2 px 以内、91% 在 5 px 以内；差的 9 帧是有人进画面时骨架晚了一两帧出现。快 35%（M2 Max 上 306 对 469 ms/帧）。`--pose-detect-every 1` 关掉它。

我们推荐 `fast`：复制走位和运镜，视频模型要的它全有——演示里的三段片，深度图交给 MiniMax H3 之前还缩到了 320×176，比这两档都小得多。同一段 294 帧的片上 full 对 fast 实测：平均差 5.3 个灰阶（满量程 255），95% 的像素差在 15 以内，边缘锐度低 4.5%。特写里在意细轮廓、显存又够，再用 `full`。运行时会打印实际用的分辨率（`model  fast: the model sees 644x364`），也写进 `--metrics`。要指定任意短边用 `--input-size`。

2026-09-12 在租来的显卡上实测；数字都在那几次运行的 `--metrics` 输出里。

## 预设

| `--target` | fps | 尺寸 | 时长 | 给谁用 |
|---|---|---|---|---|
| `seedance` | 24 | 16 的倍数，≥ 407,696 像素 | ≤ 15 秒 | Seedance 2.0 / 2.5 参考视频 |
| `h3` | 24 | 32 的倍数 | ≤ 15 秒 | MiniMax H3（参考视频或 Fun ControlNet 深度条件） |
| `wan` | 16 | 16 的倍数 | – | Wan 2.1 VACE |
| `none` | 原片 | 偶数 | – | 任何认深度视频的模型 |

`reshot --help` 列出全部参数，[使用手册](docs/USAGE.zh-CN.md)逐条解释。

## 给开发者

```python
from pathlib import Path
from reshot import RunConfig, run

run(RunConfig(input=Path("参考片.mp4"), output=Path("深度图.mp4"), target="seedance"))
```

四个细节决定了视频模型愿不愿意跟着这段片走：

- **整段只用一把尺子。** 深度对全部帧归一化一次，绝不逐帧归一化，有人从墙前走过，墙的灰度不变。逐帧归一化会让整个场景「呼吸」。
- **骨架不闪。** 逐帧的姿态估计会换人、抖关节、在阈值附近让肢体忽有忽无，视频模型把这三样全读成动作。ReShot 跨帧跟踪每个人（身体框 IoU 配对），关节一旦画上就要分数掉到远低于阈值才撤（迟滞），每个关节各过一个 One-Euro 滤波——静止时稳，出拳时跟得上。
- **按时间戳选帧。** 30 fps 转 24 fps 就真的是 24，不会以某种规律重复或丢帧让模型学了去。
- **只裁不补。** 尺寸裁到模型要的网格上，绝不补黑边——黑边会被当成远处的墙。

骨架从 Python 调：

```python
from reshot import RunConfig, run
run(RunConfig(input=Path("参考片.mp4"), output=Path("骨架.mp4"), target="h3", control="pose", keypoints=Path("骨架.json")))

from reshot.backends import get_backend
from reshot.pose.tracking import stabilise
from reshot.pose.skeleton import render_frame
clip = get_backend("dwpose").infer(frames, fps)   # frames: uint8 RGB [T, H, W, 3] → PoseClip
stabilise(clip)                                    # 跟踪 → 迟滞 → 平滑，原地改
rgb = render_frame(clip.keypoints[0], clip.scores[0], h, w, hands=True, face=False)
```

深度模型是 Video Depth Anything Small（字节跳动，CVPR 2025），以 32 帧为一个窗口重叠推理并对齐，深度不会一帧一帧地跳。12 秒 720p 一段片峰值占 3.9 GB 内存、11 GB 显存（`--input-size 364` 时 3 GB）（实测；开跑前显示的估算是拟合出来的直线，与实测误差 0.05 GB 以内），放不下会在开始之前拒绝。`fake` 后端不加载模型就能跑完整条流水线，方便你写测试。属于「用户该修」的错误都是 `ReshotError` 的子类，带退出码和具体的修法。

## 社区

- **做出片了？** 把参考片、深度图、成片和你的提示词那一行发到 [作品分享](https://github.com/maosika-ai/reshot/discussions/2)，好的会链到这里。
- **模型没照深度图走？** 到 [问答区](https://github.com/maosika-ai/reshot/discussions/categories/q-a) 贴一帧画面和提示词——多半是提示词的事。
- **出错了？** [提 issue](https://github.com/maosika-ai/reshot/issues/new/choose)，一天内回复。
- **想搭把手？** 从 [good first issue](https://github.com/maosika-ai/reshot/labels/good%20first%20issue) 开始：给别的模型加预设、翻译 README、给 H3 Fun ControlNet 剩下两个口做 HED / MLSD 输出。骨架已在 0.5.0 落地（[#6](https://github.com/maosika-ai/reshot/issues/6)）。

## 协议

Apache-2.0，默认深度模型和骨架模型（DWPose，代码和权重）也都是。模型代码 vendored 在 `reshot/third_party/`；骨架的前后处理是照 DWPose 的 ONNX 参考实现用 NumPy 重写的，[NOTICE](NOTICE) 里有署名。用在产品、流水线、服务里都行。更大的研究用深度权重（Base、Large）是 CC-BY-NC，不主动指定不会加载。

欢迎 issue 和 PR，见 [CONTRIBUTING.md](CONTRIBUTING.md)。

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
