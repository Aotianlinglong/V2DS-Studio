# models/ 里都是什么，以及它们的许可证

这个目录里的东西**不是本项目的代码**，而是运行期需要的数据文件。它们随发行版一起分发，
所以来源和条款记在这里。

## 随仓库提交的

### `budoux/`（ja / zh-hans / zh-hant .json）

BudouX 的线性分词模型，用于中文和日文的词切分。
上游：<https://github.com/google/budoux>，**Apache-2.0**。

### `cpp_pinyin/mandarin/`

拼音词典。**基于 CC-CEDICT**（MDBG 维护），
**Creative Commons Attribution-ShareAlike 4.0 International（CC BY-SA 4.0）**。
同目录下的 `License.txt` 是上游附带的原文，条款以它为准。

### `cpp_pinyin/cantonese/`

粤拼词典。**基于 CC-Canto**，
**Creative Commons Attribution-Share Alike 3.0（CC BY-SA 3.0）**。
同目录下的 `License.txt` 是上游附带的原文。

### `dictionaries/ds_cmudict-07b.txt`

CMUdict 0.7b（Carnegie Mellon University 发音词典），**BSD-2-Clause**。

### `dictionaries/ds-zh-pinyin-lite.txt`

轻量拼音词典。文件本身没有版权头，随 TIFA 的上游发布分发，**上游未声明条款**。

### `dictionaries/japanese_dict_full.txt`

日文读音词典（约 1.5 KB，是查表用的补充条目）。文件本身没有版权头，
随 TIFA 的上游发布分发，**上游未声明条款**。

### `dictionaries/jyutping_dict.txt`

粤拼补充词典。同上，**上游未声明条款**。

### `vocab.txt`

音素表，**从 TIFA 模型 `tifa-1.0-st` 内嵌的 `tifa.vocab.json` 提取**，
即对齐器实际使用的那 220 个符号（en 42 / ja 40 / yue 70 / zh 65）。
它是对模型文件的描述，不是独立创作。

## 下载的（不在仓库里）

### `lid.176.bin` / `lid.176.ftz`

fastText 的语言识别模型。fastText 的**代码**是 MIT，但**模型权重**是
**CC BY-SA 3.0**。所以它由 CI 下载并打进发行版，而不是提交进仓库——
125 MB 也不适合进 Git 历史。

## 关于 CC BY-SA

CC BY-SA 是**相同方式共享**：分发这些数据时要署名，且数据的再分发保持同一许可。
本项目的代码是 MPL-2.0，与这些数据是相互独立的作品；数据本身仍按其原许可分发，
署名信息在上面和各自的 `License.txt` 里。

## 这三个未声明条款的文件

`ds-zh-pinyin-lite.txt`、`japanese_dict_full.txt`、`jyutping_dict.txt` 来自 TIFA 的上游发布
（TIFA Label），那个发布里只带了 Electron / Chromium 的许可证，没有提到词典。
tifa.cpp 的 README 写的是"模型权重和词典保持各自的许可证 —— 见上游发布"。

**也就是说：这三个文件的条款目前无从确认。** 如果上游有明确说法，应当更新本节；
如果无法确认，稳妥的做法是把它们从发行版里去掉——代价是中文（`ds-zh-pinyin-lite.txt`）
和粤语的候选读音会少一部分，其余功能不受影响。
