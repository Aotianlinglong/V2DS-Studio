# coding=utf-8
"""全工程统一的目录锚点。

应用代码重构为 v2ds 包后，模型/词典/状态等数据目录仍位于软件根目录
（main.py、run.bat、g2p/、models/、asr_model/、inference/ 所在处）。
所有需要这些路径的模块统一从这里取，禁止再用 __file__ 向上猜目录。
"""
from pathlib import Path

# v2ds/paths.py → parents[0]=v2ds, parents[1]=软件根目录
ROOT_DIR = Path(__file__).resolve().parents[1]

# maxlabel_cli + 发音词典/LSTM 模型
G2P_DIR = ROOT_DIR / "g2p"
G2P_MODELS_DIR = G2P_DIR / "models"

# TIFA / FBL ONNX 模型
MODELS_DIR = ROOT_DIR / "models"

# ASR 整合模型（README 指引自行下载放置）
ASR_MODEL_DIR = ROOT_DIR / "asr_model"

# VST3 面板状态
VST_STATE_DIR = ROOT_DIR / ".vst_states"
