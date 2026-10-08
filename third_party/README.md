# third_party/ —— 随本项目分发的第三方二进制组件许可

本目录存放**随 V2DS Studio 二进制文件一起再分发**的组件所要求携带的许可证文本。
完整来源清单（含数据文件、Python 依赖、模型权重）见根目录 [CREDITS.md](../CREDITS.md)。

## 通过 Release「binaries 包」分发的二进制（不提交进 Git 仓库）

| 文件 | 来源 | 许可证 | 许可文本 |
|---|---|---|---|
| `g2p/maxlabel_cli.exe` | [KakaruHayate/MaxLabel](https://github.com/KakaruHayate/MaxLabel) 的未改动命令行构建版 | MPL-2.0 | [MPL-2.0.txt](MPL-2.0.txt)，源码见上游仓库 |
| `g2p/concrt140.dll`、`msvcp140*.dll`、`vcruntime140*.dll` | Microsoft Visual C++ 2015–2022 Redistributable 运行库（未改动） | 微软 Visual C++ 可再发行条款，允许随应用分发 | <https://aka.ms/vs/17/license/2022_RTM_Redistribution.rtf> |
| `inference/qwen3asr_dml/bin/` 下 `ggml*.dll`、`llama*.dll`、`mtmd.dll` | [llama.cpp](https://github.com/ggml-org/llama.cpp) 的未改动预编译版（Qwen3-ASR 的 GGUF LLM 解码后端） | MIT | [MIT-llama.cpp.txt](MIT-llama.cpp.txt) |
| `inference/qwen3asr_dml/bin/libomp140.x86_64.dll` | LLVM OpenMP 运行库（LLVM 项目，随工具链再分发） | Apache-2.0 with LLVM Exceptions | <https://releases.llvm.org/14.0.0/LICENSE.TXT> |
| `fx/bin/deep-filter.exe` | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) 的未改动预编译版 | MIT OR Apache-2.0 | [LICENSE-DeepFilterNet.txt](LICENSE-DeepFilterNet.txt) |

## 说明

- `maxlabel_cli.exe` 是 MPL-2.0 作品的可执行形式。MPL-2.0 允许在聚合产品中分发
  未修改的二进制，前提是保留许可证并提供其源码位置——源码即上游 MaxLabel 仓库，
  MPL-2.0 全文见 [MPL-2.0.txt](MPL-2.0.txt)。本项目自身的代码不受 MPL 传染。
- 以上均为**未改动的官方构建版**；如自行替换为其他版本，请以该版本随附的许可证为准。
- 模型权重（Qwen3-ASR / TIFA / FBL）不随仓库或 binaries 包分发，其许可见
  [CREDITS.md](../CREDITS.md) 的「模型权重」一节。
