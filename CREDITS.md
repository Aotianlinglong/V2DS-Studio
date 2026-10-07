# Credits / 致谢与第三方许可

V2DS Studio 站在以下开源项目的肩膀上。没有它们就没有这个工具，在此向所有作者致谢。

本项目自身代码以 **Apache License 2.0** 发布（见 [LICENSE](LICENSE)）。下列第三方组件
保持各自原许可证；使用本项目即表示你同时接受各组件许可证的条款。

## 一、代码（本仓库中包含改编/复用代码）

| 项目 | 上游 | 许可证 | 在本项目中的使用方式 |
|---|---|---|---|
| Vocal2Midi | <https://github.com/Xiantaidu/Vocal2Midi> | Apache-2.0 | `inference/`（Qwen3-ASR DML 推理、RomajiASR、ASR API）改编自该项目，详见 [inference/NOTICE.txt](inference/NOTICE.txt) |
| MaxLabel | <https://github.com/KakaruHayate/MaxLabel> | MPL-2.0 | PFML 标注格式的事实标准；PFML 编辑 UI 的设计灵感来源；随包分发未修改的 `g2p/maxlabel_cli.exe`（源码见上游仓库，许可证见 [third_party/MPL-2.0.txt](third_party/MPL-2.0.txt)） |
| TIFA | <https://github.com/openvpi/TIFA> | MIT（代码） | 对齐管线设计与 TextGrid 三层（texts/words/phones）输出格式的对齐标准；G2P 配置与预处理的参照标准 |
| audio-slicer | <https://github.com/openvpi/audio-slicer> | MIT | `audio_slicer.py` 逐行改编自上游 `slicer2.py`（即 dataset-tools AudioSlicer 的官方 Python 版），许可证见 [third_party/MIT.txt](third_party/MIT.txt) |
| dataset-tools | <https://github.com/openvpi/dataset-tools> | Apache-2.0 | 切片参数与切片行为的兼容基准 |
| g2pflow | <https://github.com/openvpi/g2pflow> | MIT | 通过 pip 依赖使用（G2P 管道），未包含其源码 |
| llama.cpp | <https://github.com/ggml-org/llama.cpp> | MIT | `inference/qwen3asr_dml/gguf/` 改编自其 gguf 工具；其原生运行库 DLL 见下文「二进制组件」 |
| cpp-pinyin / cpp-kana | <https://github.com/wolfgitpr/cpp-pinyin> | Apache-2.0 | `pinyin_engine/` 的多音字消歧逻辑参照其算法移植 |

## 二、二进制组件（随完整包 / Release binaries 包分发，不在 Git 中）

| 组件 | 上游 | 许可证 | 说明 |
|---|---|---|---|
| ggml / llama 运行库 (`ggml*.dll`、`llama*.dll` 等) | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Qwen3-ASR GGUF 解码后端，未改动官方构建，[third_party/MIT-llama.cpp.txt](third_party/MIT-llama.cpp.txt) |
| DeepFilterNet (`deep-filter.exe`) | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT OR Apache-2.0 | 降噪效果器调用的未改动预编译版，[third_party/LICENSE-DeepFilterNet.txt](third_party/LICENSE-DeepFilterNet.txt) |
| maxlabel_cli.exe | [MaxLabel](https://github.com/KakaruHayate/MaxLabel) | MPL-2.0 | 随 Release binaries 包分发（不提交进 Git），见 [third_party/MPL-2.0.txt](third_party/MPL-2.0.txt) |
| Visual C++ 运行库 (`msvcp140*.dll`、`vcruntime140*.dll`、`concrt140.dll`) | Microsoft | Visual C++ Redist 可再发行条款 | 随 Release binaries 包分发（不提交进 Git），允许随应用再分发 |
| LLVM OpenMP (`libomp140.x86_64.dll`) | [LLVM](https://releases.llvm.org/14.0.0/LICENSE.TXT) | Apache-2.0 with LLVM Exceptions | ggml CPU 后端运行依赖 |

## 三、数据文件（词典 / 模型表，随仓库提交）

| 文件 | 来源 | 许可证 |
|---|---|---|
| `g2p/models/budoux/*.json` | [google/budoux](https://github.com/google/budoux) | Apache-2.0 |
| `g2p/models/cpp_pinyin/mandarin/`、`pinyin_engine/dicts/mandarin/` | [CC-CEDICT](https://www.mdbg.net/chinese/dictionary?page=cc-cedict)（MDBG） | CC BY-SA 4.0，各目录内附 `License.txt` 原文 |
| `g2p/models/cpp_pinyin/cantonese/`、`pinyin_engine/dicts/cantonese/` | CC-Canto | CC BY-SA 3.0，各目录内附 `License.txt` 原文 |
| `g2p/models/dictionaries/ds_cmudict-07b.txt` | [CMUdict 0.7b](https://github.com/cmusphinx/cmudict)，Carnegie Mellon University | BSD-2-Clause |
| `g2p/models/dictionaries/ds-zh-pinyin-lite.txt`、`japanese_dict_full.txt`、`jyutping_dict.txt` | 随 TIFA 上游发布分发 | **上游未声明条款**，详见 [g2p/models/README.md](g2p/models/README.md) 中的说明；如权利方有异议将立即移除 |
| `g2p/models/vocab.txt` | 从 TIFA-1.0-ST 模型内嵌词表提取 | MIT（随 TIFA）；是对模型的描述性数据 |

> CC BY-SA 为「相同方式共享」许可：这些数据可自由再分发，但需保留署名且不得更换许可。
> 它们与本项目代码是相互独立的作品，不影响本项目代码的 Apache-2.0 许可。

## 四、模型权重（不随仓库分发，按 README 指引单独下载）

| 模型 | 上游 | 许可证 / 使用条款 |
|---|---|---|
| Qwen3-ASR-1.7B 权重 | **阿里 Qwen 团队**：[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（[ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) / [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)） | **Apache-2.0**（Qwen3-ASR 官方发布许可）。权重请从官方渠道下载，本项目不转发 |
| 上述权重的 DirectML 推理整合构建（encoder ONNX + LLM GGUF） | [Vocal2Midi](https://github.com/Xiantaidu/Vocal2Midi) 原创转换/打包，底层推理栈为 ONNX Runtime + [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT） | 随 Vocal2Midi 的发布条款（Apache-2.0）；获取方式见 Vocal2Midi 仓库 |
| TIFA-1.0-ST（五子图 ONNX 转换版） | [openvpi/TIFA](https://github.com/openvpi/TIFA/releases/tag/v1.0.0) | MIT（上游未对权重另设限制）；ONNX 转换版由本项目提供 |
| FoxBreatheLabeler（ONNX 转换版） | [autumn-DL/FoxBreatheLabeler](https://github.com/autumn-DL/FoxBreatheLabeler/releases) | **双许可：个人非商业使用 Apache-2.0；商业用途须遵守 AGPLv3**。商用前请阅读上游 LICENSE |

## 五、运行时依赖（pip，不随源码分发）

`requirements.txt` 中的第三方 Python 包保持各自许可证，包括但不限于：
PyQt5（**GPL-3.0** 或 Riverbank 商业许可）、onnxruntime-directml（MIT）、
numpy/scipy（BSD）、soundfile/libsndfile（BSD-3/LGPL）、librosa（ISC）、
matplotlib（PSF/BSD）、pedalboard（Apache-2.0）、dawdreamer（MIT）、
fugashi（MIT）、unidic（GPL-2.0/LGPL-2.1/BSD，词典本体）。分发整合包时请一并遵守上述条款；
本项目源码开放即满足 PyQt5 GPL 条款对源码可得性的要求。

## 六、免责声明

- V2DS Studio 是非官方的社区工具，与上述项目的维护方无隶属/背书关系。
- 文中出现的 Qwen、TIFA、DiffSinger、DeepFilterNet 等名称归各自权利人所有。
- 如任何署名或许可信息有误，请通过 Issue 指出，我们会第一时间修正。
