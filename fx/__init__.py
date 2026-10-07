# coding=utf-8
"""V2DS Studio 音频处理插件包（降噪 / 去口水音 / 去喷麦）。

设计原则：
- 每个模块只暴露 process(wave: np.ndarray, sr: int, **params) -> np.ndarray；
  输入输出等长、float32 单声道，算法完全独立，不依赖本项目其它模块。
- 插件可整体缺失：上层用 try/except 导入，缺依赖时菜单自动置灰，不影响本体。
"""
