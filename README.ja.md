<h1 align="center">ReShot</h1>

<p align="center"><b>役者ではなく、ショットをコピーする。</b></p>

<p align="center">ReShot は参照動画を深度マップに変換し、Seedance や MiniMax H3 がその振り付けとカメラワークを再現できるようにします — 登場人物はあなた自身のキャラクターに置き換えられます。</p>

<p align="center"><img src="docs/demo-fight.gif" width="720" alt="a fight scene, its depth map, and three new takes generated from it"></p>

<p align="center"><sub>上：参照動画とその深度マップ。下：その深度マップから生成された3つのテイク — 2人の女性と1匹のウサギ。同じ動き、同じカメラ。 <a href="docs/demo-fight.mp4">フル解像度クリップ</a>。</sub></p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">中文</a> · <a href="docs/USAGE.md">ユーザーガイド</a> · <a href="https://github.com/maosika-ai/ComfyUI-ReShot">ComfyUI ノード</a> · <a href="https://huggingface.co/spaces/maosika/reshot">Hugging Face</a> · <a href="https://github.com/maosika-ai/reshot/discussions">コミュニティ</a> · <a href="CHANGELOG.md">変更履歴</a></p>

<p align="center">

<a href="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml/badge.svg" alt="ci"></a>

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>

<img src="https://img.shields.io/badge/runs%20on-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">

</p>

---

## 問題

自分の AI 動画で使いたい戦闘、ダンス、またはカメラワークが完璧に入ったクリップがあります。それを再現する方法は2つありますが、どちらもうまくいきません。

- **動画モデルにクリップを参照として与える。** 動きだけでなく、顔、服、映像の見た目までコピーしてしまいます。クリップに実在の人物が映っている場合、プラットフォームのコンテンツチェックによって完全に拒否されることもあります。

- **動きを言葉で説明する。** 「彼女は壁を蹴って、パイプをつかみ、大男を肩越しに投げる」— モデルは毎回違う戦闘シーンを生成し、カメラも指示どおりには動きません。

## ReShot がすること

ReShot は `.mp4` を入力として受け取り、`.mp4` を出力します。出力は **深度マップ動画**です。各フレームはグレースケールで、近いものは白く、遠いものは黒くなります。誰がどこに立っているか、人物の相対的な大きさ、動き方、カメラの動きは保持します。一方で、顔、服、照明、スタイルは取り除きます。

<p align="center"><img src="docs/img/step1_reference.jpg" width="360" alt="reference frame"> <img src="docs/img/step2_depth.jpg" width="360" alt="the same frame as a depth map"></p>

このグレーの動画を参照として動画モデルに渡し、プロンプトで人物と映像の見た目を指定します。モデルは動画から動きを取得し、それ以外はあなたの言葉から決定します。

技術的には、単眼動画深度推定です。モデルは各フレームの相対逆深度を予測し、ReShot はそれをクリップ全体で一度だけ正規化して8ビットのグレースケール（近い = 白）に変換し、標準的な深度マップ動画としてエンコードします。

## 使い方

デモの3つのテイクは、すべてこの方法で作成されました。使用したすべてのファイルはこのリポジトリに含まれているため、同じ手順を再現できます。

### 1. インストール

**インストール不要：** [Colab でノートブックを開く](https://colab.research.google.com/github/maosika-ai/reshot/blob/main/examples/reshot_colab.ipynb) — 無料 GPU を使い、クリップをアップロードして `depth.mp4` をダウンロードできます。 <a href="https://colab.research.google.com/github/maosika-ai/reshot/blob/main/examples/reshot_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>

```bash

pip install reshot        # needs ffmpeg on PATH

```

ffmpeg がありませんか？ `pip install "reshot[ffmpeg]"` を使えば ffmpeg も含まれます。Python 環境もまったくセットアップしていませんか？ [uv](https://docs.astral.sh/uv/) なら1行ですべて実行できます：`uvx --from "reshot[ffmpeg]" reshot reference.mp4 -o depth.mp4 --target seedance`。モデルの重み（111 MB）は初回実行時にダウンロードされます。

### または、AI コーディングツールに任せる

Claude Code、Codex、Cursor、またはその他の AI エージェントを使っていますか？以下を貼り付ければ、インストール、GPU の確認、テスト用クリップの実行まで行ってくれます：

```

Install ReShot (the "reshot" package on PyPI) on this machine and get it working.

Read https://raw.githubusercontent.com/maosika-ai/reshot/main/docs/AGENT_INSTALL.md and follow it step by step:

detect the OS and GPU, install the right PyTorch build first, then `pip install "reshot[ffmpeg]"`,

set HF_ENDPOINT=https://hf-mirror.com if I'm in China, run the fake-backend smoke test, then a real run

on a short clip with --metrics and show me the numbers. Don't say it's done until depth.mp4 exists.

```

上のプロンプトは、AI エージェントに OS と GPU を検出させ、適切な PyTorch をインストールし、ReShot のセットアップとテストを完了させるためのものです。中国にいる場合のミラー設定、fake バックエンドによるスモークテスト、実際の動画での `--metrics` を使ったテストまで含まれています。

### 2. 深度マップを作成する

**最も簡単 — Web ページ。** 何も引数を付けずに `reshot` と入力すると、ブラウザでページが開きます（すべての処理はあなたのコンピューター上で行われます）。クリップをドロップし、対象モデルを選び、**Make depth map** をクリックします。参照動画と深度マップが並んで表示され、各種数値、ダウンロードボタン、そして Seedance または MiniMax H3 に貼り付けるプロンプト行が表示されます。

<p align="center"><img src="docs/img/web_ui_en.jpg" width="720" alt="the ReShot web page: reference clip and depth map side by side, target and quality on the right"></p>

```bash

reshot                     # opens http://127.0.0.1:8765 — results land in ~/ReShot

reshot web --port 9000 --out ./depth --no-browser     # options, if you want them

```

**またはコマンドライン**。スクリプトやフォルダ処理向けです：

```bash

reshot reference.mp4 -o depth.mp4 --target seedance

```

`--target seedance` は、24 fps、H.264、16 の倍数のフレームサイズ、最低 407,696 ピクセル、最大15秒に設定します。これは Seedance API の参照動画ルールに合わせたものです。MiniMax H3 では `--target h3`（32 の倍数）を使用します。RTX 4090 では12秒のクリップに約20秒、MacBook では数分かかります。フォルダ全体を一度に処理する場合も、モデルのロードは1回だけです：`reshot clips/*.mp4 -o depth/ --target seedance`。

### 3. 動画モデルに渡す

**Seedance 2.0 / 2.5。** `depth.mp4` を参照動画としてアップロードします。プロンプトでその動画を指定し、人物と映像の見た目を説明します：

```

参考@视频1的动作与运镜，顺序与视频保持一致。

一名穿深绿色丝绒旗袍的女子在狭窄的金属走廊里与三名黑衣守卫搏斗，冷蓝走廊光，红色警示灯，电影感。

```

上のプロンプトは、「動画1の動きとカメラワークを参照し、順序も動画と一致させる」ことを指定したうえで、人物、衣装、場所、照明、雰囲気を定義しています。

**ComfyUI。** [ComfyUI-ReShot](https://github.com/maosika-ai/ComfyUI-ReShot) をインストールし、Load Video と使用するモデルの間に _ReShot Depth Video_ ノードを配置するだけです。コマンドラインは必要ありません。

**MiniMax H3。** `depth.mp4` を `<Video 1>` として添付します。特定の顔を使いたい場合は、キャラクターシートを `<Picture 1>` として添付します。以下はデモで使用した3つのシートです：

<p align="center"><img src="docs/img/step3_sheets.jpg" width="720" alt="the three character sheets used as Picture 1"></p>

MiniMax H3 では、プロンプトを固定された6セクション形式で記述します。3つすべてのテイクの完全なプロンプトは [`docs/prompts/`](docs/prompts/) にあります。重要なのは、`<Video 1>` をどのように定義するか、そして何を転送してよいかを明示する部分です：

```

<Subject 3> is the fight choreography and camera movement shown in <Video 1>, a grey depth map

in which near objects are white and far objects are black: one fighter leans on a corridor wall

in close-up, kicks off it to tear down a pipe, fights several opponents, is grabbed from behind

by the largest and throws him, slams the last one into a wall panel, wipes the mouth in close-up,

then walks away through a door past the fallen opponents.

<Subject 3>: attribute_transfer - every action, position, timing and camera move of <Video 1>

is transferred onto <Subject 1> and <Subject 2>; its grey depth look is not transferred.

```

ここには重要な点が2つあります。**グレーのクリップで何が起きているかを言葉でも説明すること** — プロンプトでグレーの塊が何をしているのかを説明すると、モデルは深度マップをはるかによく理解します。**グレーの見た目自体はコピーしないよう明示すること** — そうしないと、グレーの映像がそのまま生成されることがあります。

### 4. 出力されるもの

<p align="center"><img src="docs/img/step4_takes.jpg" width="720" alt="the three takes"></p>

左から順に、ブロンズの鎧を着た [Jiang Xue](docs/prompts/take1_jiangxue_armor.txt)、緑のベルベットのチャイナドレスを着た [Su Wan](docs/prompts/take2_suwan_qipao.txt)、そして3Dアニメーション映画として、オオカミ、トラ、クマと戦う [rabbit boxer](docs/prompts/take3_rabbit_boxer.txt) です。同じ6つのショット、同じ冒頭のクローズアップ、同じ最後のドアから歩き去る動きです。参照クリップ自体は Seedance 2.0 のテキストから動画生成（864×496、12秒）で作られ、各テイクは MiniMax H3 で生成されたため、どの段階でも実在人物の肖像は使用されていません。

## 何が転送され、何が転送されないか

**転送されるもの：** 誰がどこに立っているか、人物同士の相対的な大きさ、すべての動きとそのタイミング、カット、そしてカメラワーク — プッシュイン、トラッキング、手持ちカメラの揺れ。

**転送されないもの：** 顔（キャラクターシートを使用）、服、照明、色、細かい小道具、そして手より小さいもの。これらはプロンプトと参照画像から決まります。

デモ制作で学んだこと：

- **参照動画は15秒以内にする**（Seedance の制限）。ReShot を実行する前に、使いたいショットだけに動画をカットしてください。動画に含まれるものはすべてコピーされます。最後の退屈な部分も含まれます。

- **MiniMax H3 では、深度マップを小さくする：`--target h3 --max-res 320`**（16:9 のクリップでは 320×176）。フルサイズのグレーのシルエットは、生成するキャラクターの顔の形を参照人物の顔に引っ張り始めることがあります。小さい深度マップなら、形状ではなく動きを伝えられます。

- **種を変えても、サイズ比は維持する。** クマのテイクがうまく機能するのは、プロンプトでクマを「ウサギの約1.3倍、決して1.5倍を超えない」と指定しているからです。深度マップにはすでに誰が大きいかという情報があります。プロンプトでそれと矛盾させてはいけません。

- **エキストラにはシンプルな服を使い、ロゴを入れない。** プロンプトで指定していない部分を、モデルは文字やバッジで埋めることがあります。

## 必要な環境

|                                                | 動作                                    | 実測                                                                                     |
| ---------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------- |
| **NVIDIA、8 GB以上**（RTX 3070、4060、4090 …） | はい                                    | デフォルトの `--quality fast`：VRAM 3 GB、3080 Ti で 34 ms/frame                         |
| **NVIDIA、12 GB以上**                          | はい。`--quality full` も使用可能       | full：3080 Ti で VRAM 11 GB、83 ms/frame、4090 で 62 ms/frame                            |
| **Apple Silicon**                              | はい                                    | デフォルトの `fast`：M2 Max で 46 ms/frame、12秒クリップで約18秒                         |
| **CPUのみ**                                    | はい、低速                              | 約1.8 s/frame                                                                            |
| **ホストRAM**                                  | 16 GBで720pの約27秒までのクリップに対応 | peak = 2 GB + 720p 1秒あたり224 MB。クリップが収まらない場合、ツールは開始前に拒否します |

**`--quality` — モデルが処理する解像度。** 出力動画のサイズは常に元動画と同じです。ここで設定するのは、深度モデルが見る画像のサイズです。

| `--quality`          | モデルが見るサイズ（16:9 · 9:16 · 4:3） | VRAM    | 速度                 | ディテール                                                           |
| -------------------- | --------------------------------------- | ------- | -------------------- | -------------------------------------------------------------------- |
| `fast`（デフォルト） | 644×364 · 364×644 · 490×364             | 約3 GB  | 3080 Tiで34 ms/frame | 大きな形状はfullとほぼ同じ。細い髪の毛などは頬と一体化することがある |
| `full`               | 924×518 · 518×924 · 686×518             | 約11 GB | 3080 Tiで83 ms/frame | 細かいシルエットがより鮮明                                           |

Web ページにも **Quality** として同じ2つの選択肢があります。コマンドラインでは `--quality` を使用します。

`fast` を推奨します。人物配置やカメラワークをコピーするだけなら、動画モデルに必要なのはそれで十分です — デモのテイクは MiniMax H3 に渡す前に 320×176 まで縮小された深度マップから作られており、どちらの設定よりもはるかに低い解像度です。同じ294フレームのクリップで full と fast を比較した実測値は、平均差 255階調中5.3、95%のピクセルが差15以内、エッジエネルギー −4.5% でした。細いシルエットが重要なクローズアップで、十分な VRAM がある場合は `full` を使用してください。実行時には使用した正確な解像度（`model  fast: the model sees 644x364`）が表示され、`--metrics` にも記録されます。上級者は `--input-size` で任意の短辺サイズを設定できます。

2026-09-12 にレンタルGPUで検証済みです。数値は各実行の `--metrics` 出力に記録されています。

## プリセット

| `--target` | fps    | フレームサイズ    | 長さ   | 用途                                              |
| ---------- | ------ | ----------------- | ------ | ------------------------------------------------- |
| `seedance` | 24     | ×16、≥ 407,696 px | ≤ 15 s | Seedance 2.0 / 2.5 の参照動画                     |
| `h3`       | 24     | ×32               | ≤ 15 s | MiniMax H3（参照動画または Fun ControlNet depth） |
| `wan`      | 16     | ×16               | –      | Wan 2.1 VACE                                      |
| `none`     | 元動画 | 偶数              | –      | 深度動画を読み込めるその他のツール                |

`reshot --help` ですべてのオプションを確認できます。[ユーザーガイド](docs/USAGE.md) ではそれらを詳しく説明しています。

## 開発者向け

```python

from pathlib import Path

from reshot import RunConfig, run

run(RunConfig(input=Path("reference.mp4"), output=Path("depth.mp4"), target="seedance"))

```

動画モデルが実際に追従できる出力にするため、3つの重要な処理があります：

- **クリップ全体で1つのスケール。** 深度はフレームごとではなく、すべてのフレームを対象に一度だけ正規化されます。そのため、誰かが壁の前を通っても壁のグレー値は変わりません。フレームごとの正規化では、シーンが「呼吸している」ように見えてしまいます。

- **タイムスタンプでフレームを選択。** 30 fps → 24 fps は本当に24 fpsになります。モデルが学習してしまうような、規則的なフレームの複製や削除は行いません。

- **パディングではなくクロップ。** フレームサイズはモデルのグリッドに合わせて切り詰められます。黒い枠は、遠くの壁として解釈されてしまう可能性があります。

モデルは Video Depth Anything Small（ByteDance、CVPR 2025）です。重なり合う32フレームのウィンドウで処理し、それらを整列させるため、フレーム間で深度がちらつきません。12秒の720pクリップでは、ホストRAMのピークは3.9 GB、VRAMは11 GB（`--input-size 364` では3 GB）です（実測値。開始前にツールが表示する推定値はフィッティングされた直線で、実測値との差は0.05 GB以内）。クリップがメモリに収まらない場合は、開始前に拒否します。`fake` バックエンドを使えば、自分のテスト用にモデルなしでパイプライン全体を実行できます。ユーザー側で修正可能なエラーは `ReshotError` のサブクラスとして扱われ、終了コードと具体的な修正方法がメッセージに表示されます。

## コミュニティ

- **何か作りましたか？** 参照動画、深度マップ、生成結果、使用したプロンプト行を [Show and tell](https://github.com/maosika-ai/reshot/discussions/2) に投稿してください。優れた作品はここからリンクされます。

- **モデルが深度マップに従わなかった？** 1フレームとプロンプトを添えて [Q&A](https://github.com/maosika-ai/reshot/discussions/categories/q-a) で質問してください。多くの場合、原因はプロンプトです。

- **何か壊れましたか？** [Issue を作成](https://github.com/maosika-ai/reshot/issues/new/choose)してください。1日以内に回答します。

- **手伝いたいですか？** [good first issue](https://github.com/maosika-ai/reshot/labels/good%20first%20issue) から始めてください。別のモデル向けプリセット、Colab ノートブック、あなたの言語の README などがあります。Pose 出力が大きなテーマです — [#6](https://github.com/maosika-ai/reshot/issues/6) を参照してください。

## ライセンス

Apache-2.0。デフォルトモデルも同じライセンスです。ベンダー提供のモデルコードは `reshot/third_party/` にあります。製品、パイプライン、サービスで使用できます。より大きな研究用途限定の重み（Base、Large）は CC-BY-NC で、明示的に指定しない限り読み込まれません。

Issue と Pull Request を歓迎します。詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## Maosika 猫斯卡について

ReShot は、**[Maosika 猫斯卡](https://www.maosika.com)**（[www.maosika.com）がオープンソースとして公開しています。Maosika](http://www.maosika.com）がオープンソースとして公開しています。Maosika) は、短編ドラマやショート動画向けのプロフェッショナルな **AI動画制作システム**です。Maosika は、1行のアイデアから完成した縦型 AI ショートドラマまで、ストーリーを制作します。エピソード形式の脚本を書き、キャラクターとシーンをデザインし、一貫性のあるキャラクターシートとシーン画像を生成し、**Seedance 2.0 / 2.5** や **MiniMax H3** などの動画モデルで各ショットをレンダリングします。各工程をデジタル専門家のチームが担当するため、かつてはスタジオが必要だったシリーズを1人でも制作できます。個人の脚本家、MCN、ショートドラマ制作会社が、Maosika を使って毎日 AI ショートドラマを制作しています。

ReShot はそのパイプラインにおける深度マップの工程です。Apache-2.0 の下で公開されているため、誰でも参照ショットのステージングとカメラワークを、自分自身の AI 生成動画にコピーできます。なぜこのツールを作ったのか、デモのテイクとプロンプトについては、**[maosika.com の ReShot 記事](https://www.maosika.com/blog/reshot-open-source-depth-motion-capture-seedance-minimax-h3)** をご覧ください。

([中文](https://www.maosika.com/blog/reshot-open-source-depth-motion-capture))。AI ショートドラマ、AI ショート動画、AI 漫画ドラマをエンドツーエンドで制作したい場合は、

**https://www.maosika.com** をご覧ください。

<p align="center"><sub>Made by <a href="https://www.maosika.com">Maosika 猫斯卡</a> · AI short drama, produced daily · <a href="https://www.maosika.com">www.maosika.com</a></sub></p>
