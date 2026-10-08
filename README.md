# V2DS Studio

**Vocal to Dataset Studio** —— 为 [DiffSinger](https://github.com/openvpi/DiffSinger) 数据集制作设计的一体化标注工作台，覆盖从原始人声到 `build dataset`（binarize）之前的全部环节：

> 切片 → ASR 识别歌词 → 歌词/语言标注 → 自动标音（G2P + 多音字消歧）→ 呼吸音检测（AP/EP）→ TIFA 强制对齐 → TextGrid / PFML 导出 + 对齐质量统计

![Platform](https://img.shields.io/badge/platform-Windows%20x64-blue) ![Python](https://img.shields.io/badge/python-3.12-blue) [![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

*English: V2DS Studio is an all-in-one labeling workbench for DiffSinger dataset creation on Windows (slicing → ASR → lyric/language annotation → G2P → breath detection → TIFA forced alignment → TextGrid/PFML export with quality statistics).*

## 功能一览

| 模块 | 说明 |
|---|---|
| 波形切片 | 波形 + 频谱视图，鼠标框选/拖动切片边界，自动吸附静音段（算法与 openvpi/dataset-tools 一致） |
| 播放控制 | 选中片段循环播放，支持 0.5~2.0 倍保调变速（相位声码器，音调不变），可经 VST3 实时监听 |
| ASR 识别 | Qwen3-ASR（ONNX encoder + GGUF LLM，DirectML 推理，A/N/I 卡通用），支持多语言混合识别 |
| 歌词标注 | 标记列表管理，逐片段填写歌词、设置语言（zh / ja / en / yue），支持片段级多语言 |
| 自动标音 | g2pflow 管道（与 TIFA/tifa.cpp 同源），拼音/粤拼/日语 MeCab/英语 LSTM，多音字词典消歧 |
| 呼吸检测 | FoxBreatheLabeler ONNX 模型检测呼吸音，自动在词间插入 AP、句尾插入 EP（可开关） |
| 强制对齐 | TIFA ONNX 五子图管线（spectrogram / prepare / score / select / model） |
| 导出 | TextGrid（texts / words / phones 三层，对齐原 TIFA 输出）+ PFML，含对齐质量统计图 |
| 效果器 | VST3 面板（降噪 DeepFilterNet / 去口水音 / 去喷麦） |

## 下载与安装

### 方式一：完整发行包（推荐普通用户）

到 [Releases](../../releases) 页面下载：

1. `V2DS-Studio_x.x.x_win64.7z` —— 已捆绑 Python 运行环境与全部原生二进制，解压即用
   （文件较大，如下载入口在 Release 说明中的网盘链接，以 Release 页面为准）；
2. FBL、TIFA 两个模型包（见下表），按「放置位置」解压；
3. Qwen3-ASR 模型请按[下方说明](#qwen3-asr-17b-dml)从指定地址下载，放到软件根目录的 `asr_model/` 下；
4. 双击 `run.bat` 启动。

### 方式二：源码运行（开发者）

需要 Windows 10/11 x64、Python 3.12：

```bash
git clone https://github.com/Aotianlinglong/V2DS-Studio.git
cd V2DS-Studio
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m unidic download
```

再下载 Release 中的 `V2DS-Studio_binaries_pack.zip`（约 50 MB，含 ggml/llama 运行库、
deep-filter 可执行文件、maxlabel_cli 与 VC++ 运行时），**解压到仓库根目录**
（包内 `inference/`、`fx/`、`g2p/` 与源码目录合并）。

```bash
python main.py
```

> ONNX Runtime 使用 **onnxruntime-directml**；无独显也可改装 `onnxruntime`（CPU）。
> 日语分词依赖 fugashi + 完整版 UniDic（已在 requirements 中），安装后需执行一次
> `python -m unidic download` 拉取词典。

## 模型下载与放置（重要）

FBL、TIFA 的 ONNX 转换包在本项目 [Releases](../../releases) 提供；
**Qwen3-ASR 权重不随本项目转发**，请按下方链接单独下载。模型放到软件根目录的
`asr_model/` 即可被自动识别；如需放在其他位置，可在 `config.json` 中设置
`onnx_asr_model_path` 为模型目录（路径失效时软件也会自动重新探测）。
各模型的许可证见 [CREDITS.md](CREDITS.md#四模型权重不随仓库分发按-readme-指引单独下载)。

### Qwen3-ASR-1.7B-DML

模型权重本身是**阿里 Qwen 团队的 [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR)（Apache-2.0）**。
本软件需要的 **DirectML 推理转换版**（ONNX encoder + llama.cpp GGUF 解码，
A / N / I 显卡通用）由 **[Vocal2Midi](https://github.com/Xiantaidu/Vocal2Midi) 项目开发者转换发布**，
直接在此下载：

- **DML 转换版（本软件直接可用）**：<https://huggingface.co/xian-taidu/Qwen3-ASR-1.7B-DML>
- 原版权重出处（仅供参考）：[QwenLM/Qwen3-ASR (GitHub)](https://github.com/QwenLM/Qwen3-ASR)、
  [ModelScope（国内推荐）](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)、
  [Qwen/Qwen3-ASR-1.7B (Hugging Face)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

下载后把模型文件放到软件根目录的 `asr_model/` 下，核心文件为：

```
asr_model/
├─ qwen3_asr_encoder_frontend.fp16.onnx
├─ qwen3_asr_encoder_backend.fp16.onnx
└─ qwen3_asr_llm.f16.gguf
```

注意事项：

- 具体文件**以该 HF 仓库实际内容为准**；除 fp16 / f16 组合外，软件同样支持
  `*.int4.onnx` encoder 与 `qwen3_asr_llm.q4_k.gguf` 的量化组合（见 `inference/qwen3asr_dml/runtime.py`）；
  tokenizer 与词表已内嵌于 GGUF，无需额外的 `tokenizer.json` / `embed_tokens.bin`。
- 国内无法直连 Hugging Face 时，可使用 HF 镜像站或自行设置代理下载。
- DML 转换版仅适用于本软件的 DirectML 推理流程；如需原版 Safetensors 权重请走千问官方渠道。
- 权重版权归阿里 Qwen 团队；转换构建的归属与许可见 [CREDITS.md](CREDITS.md)。

### 其他模型（本项目 Release 提供）

| 模型 | 包名 | 放置位置 |
|---|---|---|
| TIFA 对齐模型（由 [openvpi/TIFA v1.0.0](https://github.com/openvpi/TIFA/releases/tag/v1.0.0) 转换） | `V2DS-Studio_tifa_pack.zip` | 解压到软件根目录，形成 `models/tifa/`（spectrogram/prepare/score/select/model.onnx + config.json + vocabulary.json） |
| FBL 呼吸模型（转换自 [FoxBreatheLabeler](https://github.com/autumn-DL/FoxBreatheLabeler/releases)） | `V2DS-Studio_fbl_pack.zip` | 解压到软件根目录，形成 `models/fbl/`（model.onnx + config.yaml） |

> **FBL 模型为双许可**：个人非商业使用 = Apache-2.0；商业用途 = AGPLv3，商用请先阅读上游 LICENSE。

## 目录结构

```
V2DS-Studio/
├─ main.py                  # 入口（先加载 ONNX Runtime，再启动 PyQt5）
├─ run.bat                  # 启动脚本（完整包用捆绑 python\）
├─ requirements.txt         # 源码运行依赖
├─ v2ds/                    # 应用代码包
│  ├─ paths.py              # 统一路径锚点（根目录 / 模型 / g2p / .vst_states）
│  ├─ g2p.py                # G2P 管道（多语言候选 / 注音 / 日语 MeCab 分词）
│  ├─ core/                 # 配置、工程、PFML、字符 ↔ 词格映射、char_id
│  ├─ audio/                # 播放、音频加载、静音切片、波形频谱视图
│  ├─ asr/                  # ASR 引擎 / 子进程 / 服务封装
│  ├─ align/                # TIFA ONNX 对齐管线 + TextGrid/PFML、呼吸插入
│  └─ ui/                   # 主窗口、对话框、导出、PFML 按键行、VST 面板
├─ g2p/                     # maxlabel_cli 与词典数据（models/，二进制由 binaries 包提供）
├─ pinyin_engine/           # 多音字词典消歧（cpp-pinyin 算法移植）
├─ inference/               # Qwen3-ASR DML 推理（改编自 Vocal2Midi，Apache-2.0）
├─ fx/                      # 效果器（deep-filter.exe / VST 扫描由 binaries 包提供）
├─ third_party/             # 随二进制分发的第三方许可证文本
├─ CREDITS.md               # 全部第三方组件与许可证清单
├─ models/                  # 【自行放置】FBL / TIFA 模型
└─ asr_model/               # 【自行放置】Qwen3-ASR 模型
```

## 工作流

1. 加载 wav → 框选切片生成标记（切片行为与 dataset-tools 一致）
2. ASR 识别填词（可多语言混合），手动修正文本与语言标记
3. 自动标音：发音候选 / 多音字消歧 / 插入音素（AP、EP、cl、n、ja/cl 等，可自定义快捷音素及前置/后置规则）
4. 呼吸检测：FBL 模型自动插入 AP/EP
5. TIFA 对齐 → 导出 TextGrid + PFML + 对齐质量统计
6. 产出的 TextGrid 可直接进入 DiffSinger 的 build dataset（binarize）流程

## 反馈与贡献

欢迎提 Issue，为了高效排查，请附上：**操作系统、软件版本（完整包/源码）、复现步骤、
报错截图或日志**。功能建议也欢迎开 Issue 讨论；提交 PR 请保持代码风格与现有文件一致。

## 致谢与许可证

本项目得到了以下开源项目的帮助：[Vocal2Midi](https://github.com/Xiantaidu/Vocal2Midi)、
[MaxLabel](https://github.com/KakaruHayate/MaxLabel)、[TIFA](https://github.com/openvpi/TIFA)、
[FoxBreatheLabeler](https://github.com/autumn-DL/FoxBreatheLabeler)、
[dataset-tools](https://github.com/openvpi/dataset-tools)、
[audio-slicer](https://github.com/openvpi/audio-slicer)、[g2pflow](https://github.com/openvpi/g2pflow)、
[llama.cpp](https://github.com/ggml-org/llama.cpp)、[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
以及 BudouX、CC-CEDICT、CC-Canto、CMUdict 等数据项目。完整归属与许可证清单见
[CREDITS.md](CREDITS.md)。

本项目自身代码以 **Apache License 2.0** 发布，详见 [LICENSE](LICENSE)；
第三方组件、词典数据与模型权重保持各自上游许可证。
