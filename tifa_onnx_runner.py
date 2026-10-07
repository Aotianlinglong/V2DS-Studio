# coding=utf-8
"""TIFA + FBL ONNX 推理编排器（无 torch，用捆绑 Python 的 onnxruntime + numpy）。

职责：
  - FBL 呼吸检测：音频 → ONNX → AP 概率 → 找段 → 与对齐词边界映射 → 插入位置
  - TIFA 对齐（后续扩展）：G2P → 5 ONNX 子图串联 → 对齐结果

环境：V2DS Studio 整合包 Python，已有 onnxruntime(DML)、numpy、numba、soundfile、scipy。
"""
from __future__ import annotations
import math
import yaml
import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_FBL_DIR = _DIR / "models" / "fbl"
_FBL_ONNX = _FBL_DIR / "model.onnx"
_FBL_CFG = _FBL_DIR / "config.yaml"

# ONNX Runtime providers：优先 DirectML（通吃 N/A/核显），回退 CPU
_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]

# 抑制非致命 shape-merge 警告（score.onnx descriptors 维度推导差异）
ort.set_default_logger_severity(3)  # 3=ERROR，只输出错误及以上

# FBL session 单例（懒加载）
_fbl_sess = None


def _fbl_session():
    global _fbl_sess
    if _fbl_sess is None:
        if not _FBL_ONNX.exists():
            raise RuntimeError(f"FBL ONNX 模型不存在: {_FBL_ONNX}")
        _fbl_sess = ort.InferenceSession(str(_FBL_ONNX), providers=_PROVIDERS)
    return _fbl_sess


def _fbl_config():
    if not _FBL_CFG.exists():
        raise RuntimeError(f"FBL 配置不存在: {_FBL_CFG}")
    return yaml.safe_load(_FBL_CFG.read_text(encoding="utf-8"))


def fbl_available() -> bool:
    return _FBL_ONNX.exists() and _FBL_CFG.exists()


def fbl_find_segments(
    prob: np.ndarray,
    time_scale: float,
    *,
    threshold: float = 0.4,
    max_gap: int = 5,
    ap_dur: float = 0.08,
) -> list[tuple[float, float]]:
    """从 FBL 概率序列找 AP 段（numpy，无 torch）。"""
    ap_threshold = max(1, int(ap_dur / time_scale))
    segments = []
    start = None
    gap_count = 0
    for i in range(len(prob)):
        if prob[i] >= threshold:
            if start is None:
                start = i
            gap_count = 0
        else:
            if start is not None:
                if gap_count < max_gap:
                    gap_count += 1
                else:
                    end = i - gap_count - 1
                    if end >= start and (end - start) >= ap_threshold:
                        segments.append((start * time_scale, end * time_scale))
                    start = None
                    gap_count = 0
    if start is not None and (len(prob) - start) >= ap_threshold:
        segments.append((start * time_scale, (len(prob) - 1) * time_scale))
    return segments


def fbl_infer(wav_path: str | Path, start: float = 0.0, end: float | None = None) -> list[tuple[float, float]]:
    """对音频片段跑 FBL ONNX 推理，返回 AP 段列表 [(start_s, end_s), ...]。"""
    sess = _fbl_session()
    cfg = _fbl_config()
    sr_target = cfg["audio_sample_rate"]      # 44100
    hop = cfg["hop_size"]                     # 882
    time_scale = 1.0 / (sr_target / hop)      # 0.02

    data, sr = sf.read(str(wav_path), always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    i0 = int(round(start * sr))
    i1 = int(round(end * sr)) if end is not None else len(data)
    i0, i1 = max(0, i0), min(len(data), i1)
    seg = data[i0:i1]
    if len(seg) == 0:
        return []

    # 重采样到目标采样率
    if sr != sr_target:
        from scipy import signal
        n_target = int(len(seg) * sr_target / sr)
        seg = signal.resample(seg, n_target)

    seg = seg.astype(np.float32)[None, :]  # [1, L]
    prob = sess.run(None, {"waveform": seg})[0]  # [1, T]
    sxp = prob[0] if prob.ndim == 2 else prob[0, 0]
    return fbl_find_segments(sxp, time_scale)


def fbl_inserts_from_segments(
    segments: list[tuple[float, float]],
    word_boundaries: list[tuple[float, float]],
    text_length: int,
) -> list[dict]:
    """把 FBL 检出的 AP 段映射到字符位置 → 插入指令 [{"b": pos, "sym": "AP"/"EP", "side": "before"/"after"}]。

    规则（与旧 FBL Python 流程等价）：
      - 段中心落在哪个词间空隙 → 插在前词后（after）
      - 段中心在最后词结束之后 → EP（after 末尾）
      - 段中心在首词之前 → AP（before 首词）
    """
    if not segments:
        return []
    if not word_boundaries:
        return []

    inserts = []
    for seg_start, seg_end in segments:
        center = (seg_start + seg_end) / 2.0
        # 找 center 落在哪个词间空隙
        placed = False
        for i in range(len(word_boundaries) - 1):
            w1_end = word_boundaries[i][1]
            w2_start = word_boundaries[i + 1][0]
            if w1_end <= center < w2_start:
                # 词间空隙：插在第 i 个词后（after）
                pos = word_boundaries[i][1]  # 字符位置 = 该词结束位置
                inserts.append({"b": pos, "sym": "AP", "side": "after"})
                placed = True
                break
        if not placed:
            # 检查是否在首词之前
            if center < word_boundaries[0][0]:
                inserts.append({"b": 0, "sym": "AP", "side": "before"})
            # 检查是否在最后词之后
            elif center >= word_boundaries[-1][1]:
                inserts.append({"b": text_length, "sym": "EP", "side": "after"})
            else:
                # 落在某个词内部（理论上 FBL 只检空隙，但防边界情况）
                # 找最近的词边界
                best_i, best_dist = 0, abs(center - word_boundaries[0][1])
                for i in range(len(word_boundaries)):
                    d = abs(center - word_boundaries[i][1])
                    if d < best_dist:
                        best_dist, best_i = d, i
                inserts.append({"b": word_boundaries[best_i][1], "sym": "AP", "side": "after"})
    return inserts


# ---------------------------------------------------------------------------
# TIFA ONNX 对齐（替代 tifa_ggml_cli align）
#
# 流水线（与 TIFA 参考实现 inference/module.py 等价，纯 numpy 无 torch）：
#   音频 → spectrogram.onnx → (spec, maskT)
#   PFML → g2pflow → encode_paths（候选网格）→ prepare.onnx → (tokens, segments, mapping)
#   有歧义段 → model.onnx(打分) → score.onnx → host 整词 DP → choices
#   select.onnx → (best_tokens, best_words, best_groups, maskN)
#   model.onnx(对齐) → similarities → Viterbi(flat decode) → 帧区间 ×timestep
# 输出写成与 cpp 相同的 TextGrid（texts/phones 两层），下游解析逻辑不变。
# ---------------------------------------------------------------------------
_TIFA_DIR = _DIR / "models" / "tifa"
_TIFA_CFG = _TIFA_DIR / "config.json"
_TIFA_VOCAB = _TIFA_DIR / "vocabulary.json"

_tifa_sessions: dict = {}
_tifa_cfg_cache: dict | None = None
_tifa_vocab_cache: "_Vocabulary | None" = None

_MASK_TOKEN = 1
_SPACE_TOKEN = 2
_RESERVED = 3


def tifa_available() -> bool:
    return all((_TIFA_DIR / f"{n}.onnx").exists() for n in (
        "spectrogram", "model", "prepare", "score", "select"
    )) and _TIFA_CFG.exists() and _TIFA_VOCAB.exists()


def _tifa_session(name: str) -> "ort.InferenceSession":
    if name not in _tifa_sessions:
        _tifa_sessions[name] = ort.InferenceSession(
            str(_TIFA_DIR / f"{name}.onnx"), providers=_PROVIDERS)
    return _tifa_sessions[name]


def _tifa_config() -> dict:
    global _tifa_cfg_cache
    if _tifa_cfg_cache is None:
        import json
        _tifa_cfg_cache = json.loads(_TIFA_CFG.read_text(encoding="utf-8"))
    return _tifa_cfg_cache


class _Vocabulary:
    """vocabulary.json：{symbols: {symbol: id}}，多符号可共享 id。语言前缀解析。"""

    def __init__(self, symbol_to_id: dict):
        self.symbol_to_id = symbol_to_id
        id_to: dict[int, list] = {}
        for s, i in symbol_to_id.items():
            id_to.setdefault(i, []).append(s)
        self.id_to_symbols = id_to

    def resolve(self, symbol: str, languages=()):
        """先查裸符号，再按语言序查 lang/symbol。返回 (id, symbol) 或 None。"""
        if symbol in self.symbol_to_id:
            return self.symbol_to_id[symbol], symbol
        if "/" in symbol:
            return None
        for lang in languages:
            p = f"{lang}/{symbol}"
            if p in self.symbol_to_id:
                return self.symbol_to_id[p], p
        return None

    def label(self, token: int) -> str:
        syms = self.id_to_symbols.get(int(token))
        if not syms:
            return ""
        return syms[0]  # 完整符号（含语言前缀），主语言前缀在输出处统一剥除


def _tifa_vocabulary() -> _Vocabulary:
    global _tifa_vocab_cache
    if _tifa_vocab_cache is None:
        import json
        data = json.loads(_TIFA_VOCAB.read_text(encoding="utf-8"))
        _tifa_vocab_cache = _Vocabulary(data["symbols"])
    return _tifa_vocab_cache


# ---------------------------------------------------------------------------
# Levenshtein 多序列对齐（移植 TIFA lib/levenshtein.py）
# ---------------------------------------------------------------------------

def _align_pair(orig: list, mut: list):
    """Levenshtein DP 对齐，None 标记空隙；回溯 匹配>删除>插入。"""
    n, m = len(orig), len(mut)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if orig[i - 1] == mut[j - 1] else 1
            match_v = dp[i - 1][j - 1] + cost
            del_v = dp[i - 1][j] + 1
            ins_v = dp[i][j - 1] + 1
            if match_v <= del_v and match_v <= ins_v:
                dp[i][j] = match_v
            elif del_v <= ins_v:
                dp[i][j] = del_v
            else:
                dp[i][j] = ins_v
    al_o, al_m = [], []
    i, j = n, m
    while i > 0 or j > 0:
        match_ok = False
        if i > 0 and j > 0:
            cost = 0 if orig[i - 1] == mut[j - 1] else 1
            match_ok = dp[i][j] == dp[i - 1][j - 1] + cost
        if match_ok:
            al_o.append(orig[i - 1]); al_m.append(mut[j - 1]); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            al_o.append(orig[i - 1]); al_m.append(None); i -= 1
        else:
            al_o.append(None); al_m.append(mut[j - 1]); j -= 1
    al_o.reverse(); al_m.reverse()
    return al_o, al_m


def _consensus(rows: list) -> list:
    out = []
    for col in range(len(rows[0])):
        counts: dict = {}
        for row in rows:
            t = row[col]
            if t is not None:
                counts[t] = counts.get(t, 0) + 1
        out.append(max(counts, key=counts.get) if counts else None)
    return out


def _merge_profile(rows: list, new_path: list) -> list:
    cons = _consensus(rows)
    al_c, al_p = _align_pair(cons, new_path)
    new_rows = []
    for old in rows:
        nr = []
        k = 0
        for c_tok in al_c:
            if c_tok is not None:
                nr.append(old[k]); k += 1
            else:
                nr.append(None)
        new_rows.append(nr)
    new_rows.append(al_p)
    return new_rows


def _align_multiple_sequences(paths: list) -> list:
    if not paths:
        return []
    rows = [list(paths[0])]
    for p in paths[1:]:
        rows = _merge_profile(rows, p)
    return rows


# ---------------------------------------------------------------------------
# G2P 候选网格编码（移植 TIFA lib/g2p_encoding.py，force 模式：OOV 候选整条丢弃）
# ---------------------------------------------------------------------------

def _resolve_phoneme(phoneme: str, language, vocab: _Vocabulary, languages):
    from g2pflow import Language
    if language is Language.ANY:
        candidates = tuple(languages or ())
    elif language is not None:
        candidates = (str(language),)
    else:
        candidates = ()
    return vocab.resolve(phoneme, candidates)


def _encode_paths(g2p_words, vocab: _Vocabulary, languages):
    """G2PWord 列表 → (data, lexicon, texts)。

    data: paths [P,C] int64 / words [P] int64 / groups [P,C] int64 / candidates [W,C] bool
    """
    all_cands = []
    for word in g2p_words:
        cands = []
        for ri, reading in enumerate(word.readings):
            for path in reading.paths:
                tokens, groups, phonemes, scripts = [], [], [], []
                unknown = False
                for group in path:
                    if not group.phonemes:
                        raise ValueError("空发音组")
                    gt = []
                    for ph in group.phonemes:
                        res = _resolve_phoneme(ph, word.language, vocab, languages)
                        if res is None:
                            unknown = True
                            continue
                        tok, sym = res
                        if tok < _RESERVED:
                            raise ValueError(f"G2P 输出保留符号: {ph!r}")
                        gt.append(tok)
                        phonemes.append(sym)  # 保留 vocabulary 中的完整符号（含语言前缀）
                    if gt:
                        scripts.append(group.script)
                        tokens.extend(gt)
                        groups.extend([len(scripts)] * len(gt))
                if unknown:
                    continue  # force：含未知音素的候选整条丢弃
                cands.append({"tokens": tokens, "groups": groups,
                              "phonemes": phonemes, "scripts": scripts,
                              "reading": ri})
        all_cands.append(cands)

    profiles = []
    for cands in all_cands:
        rows = _align_multiple_sequences([c["tokens"] for c in cands])
        profile = (np.asarray([[t if t is not None else 0 for t in row] for row in rows],
                              dtype=np.int64).reshape(len(cands), -1)
                   if cands else np.zeros((0, 0), dtype=np.int64))
        profiles.append(profile)

    capacity = max(1, sum(p.shape[1] for p in profiles))
    width = max(1, max(map(len, all_cands), default=0))
    tokens = np.zeros((capacity, width), dtype=np.int64)
    groups = np.zeros_like(tokens)
    owners = np.zeros(capacity, dtype=np.int64)
    valid = np.zeros((len(all_cands), width), dtype=np.bool_)
    offset = 0
    for w, (cands, profile) in enumerate(zip(all_cands, profiles)):
        size = profile.shape[1]
        count = len(cands)
        valid[w, :count] = True
        owners[offset:offset + size] = w + 1
        tokens[offset:offset + size, :count] = profile.T
        for c, cand in enumerate(cands):
            present = profile[c] != 0
            groups[offset:offset + size, c][present] = cand["groups"]
        offset += size
    data = {"paths": tokens, "words": owners, "groups": groups, "candidates": valid}
    lexicon = [[{"reading": c["reading"], "phonemes": c["phonemes"],
                 "scripts": c["scripts"]} for c in cands]
               for cands in all_cands]
    return data, lexicon, [w.text for w in g2p_words]


# ---------------------------------------------------------------------------
# 整词候选 DP（移植 TIFA inference/scoring.py 的 host 部分，纯 numpy）
# ---------------------------------------------------------------------------

def _advance(state, pieces, choice, descriptors, lengths, costs, tails, capacity):
    segment, used = state
    score = 0.0
    for fragment in pieces:
        current = int(descriptors[fragment, 1])
        if current != segment:
            score += float(tails[segment, used])
            segment, used = current, 0
        length = int(lengths[fragment, choice])
        if used + length > capacity[segment]:
            return None
        score += float(costs[fragment, choice, used])
        used += length
    return (segment, used), score


def _select_choices(valid, descriptors, lengths, costs, tails, capacity):
    """整词 Viterbi 前向 + 回溯 + 条件分数。返回 (choices [W], scores [W,C])。

    choices 1-based（0=缺失词）；scores 为整词条件 log 分数（未选/不可行 = -inf）。
    与 TIFA scoring._select_sample 一致（确定性平局 + 反向传播分数）。
    """
    forward = [{(0, 0): 0.0}]
    ranks = {(0, 0): 0}
    parents, layers = [], []
    for w, row in enumerate(valid):
        pieces = np.flatnonzero(descriptors[:, 0] == w + 1)
        cand_ids = (np.flatnonzero(row) + 1).tolist() or [0]
        following, parent, order, edges = {}, {}, {}, []
        for source, value in forward[-1].items():
            for c in cand_ids:
                transition = (_advance(source, pieces, c - 1, descriptors,
                                       lengths, costs, tails, capacity)
                              if c > 0 else (source, 0.0))
                if transition is None:
                    continue
                target, cost = transition
                edges.append((source, target, c, cost))
                total = value + cost
                key = (ranks[source], c)
                if (target not in following or total > following[target]
                        or (total == following[target] and key < order[target])):
                    following[target] = total
                    parent[target] = (source, c)
                    order[target] = key
        if not following:
            raise ValueError("打分模板无可行整词路径")
        ranks = {state: rank for rank, state in enumerate(sorted(order, key=order.get))}
        forward.append(following)
        parents.append(parent)
        layers.append(edges)
    terminal = {state: float(tails[state[0], state[1]]) for state in forward[-1]}
    state = min(forward[-1], key=lambda x: (-(forward[-1][x] + terminal[x]), ranks[x]))
    choices = np.zeros(len(valid), dtype=np.int64)
    for i in range(len(valid) - 1, -1, -1):
        state, choices[i] = parents[i][state]

    scores = np.full(valid.shape, -np.inf, dtype=np.float64)
    backward = terminal
    for i in range(len(valid) - 1, -1, -1):
        previous = {s: -math.inf for s in forward[i]}
        for source, target, c, cost in layers[i]:
            suffix = cost + backward[target]
            previous[source] = max(previous[source], suffix)
            if c > 0:
                scores[i, c - 1] = max(scores[i, c - 1],
                                       forward[i][source] + suffix)
        backward = previous
    return choices, scores


# ---------------------------------------------------------------------------
# Viterbi flat 解码（移植 TIFA modules/decoding.py decode_alignment_flat，无 numba）
# ---------------------------------------------------------------------------

def _canonicalize_skipped(spans: np.ndarray, T: int, gap_allowed: np.ndarray):
    """把被跳过的连续 token 的退化区间锚定到第一个允许等待的空隙。"""
    N = len(spans)
    lo = 0
    while lo < N:
        if spans[lo, 0] != spans[lo, 1]:
            lo += 1
            continue
        hi = lo
        while hi + 1 < N and spans[hi + 1, 0] == spans[hi + 1, 1]:
            hi += 1
        left = spans[lo - 1, 1] if lo > 0 else 0
        right = spans[hi + 1, 0] if hi + 1 < N else T
        gap_index = lo
        while gap_index <= hi + 1 and not gap_allowed[gap_index]:
            gap_index += 1
        for i in range(lo, hi + 1):
            anchor = left if i < gap_index else right
            spans[i, 0] = anchor
            spans[i, 1] = anchor
        lo = hi + 1


def _decode_flat_single(sim: np.ndarray, skip_penalty: float,
                        gap_allowed: np.ndarray) -> np.ndarray:
    """每 token 至少发一帧或付 skip_penalty；退出/跳帧不耗帧。返回 [N,2] 帧区间。"""
    T, N = sim.shape
    penalty = np.float32(skip_penalty)
    gap = np.full(N + 1, -np.inf, dtype=np.float32)
    token = np.full(N, -np.inf, dtype=np.float32)
    gap_back = np.zeros((T + 1, N + 1), dtype=np.int8)
    token_back = np.zeros((T + 1, N), dtype=np.int8)
    gap[0] = 0.0
    for i in range(N):
        gap[i + 1] = gap[i] - penalty
        gap_back[0, i + 1] = 2  # skip

    for t in range(1, T + 1):
        row = sim[t - 1]
        # token[i] = max(token[i], gap[i]) + sim
        from_gap = gap[:N] > token
        token_back[t, from_gap] = 1
        token = np.where(from_gap, gap[:N], token) + row
        next_gap = np.full(N + 1, -np.inf, dtype=np.float32)
        # wait：消耗一帧
        np.copyto(next_gap, gap, where=gap_allowed)
        # exit（token→gap，不耗帧）与 skip（gap[i]→gap[i+1]）必须按 i 顺序执行：
        # 二者在同帧内可级联（exit 后立刻 skip、连续 skip 多个 token）。
        for i in range(N):
            if token[i] > next_gap[i + 1]:
                next_gap[i + 1] = token[i]
                gap_back[t, i + 1] = 1
            skipped = next_gap[i] - penalty
            if skipped > next_gap[i + 1]:
                next_gap[i + 1] = skipped
                gap_back[t, i + 1] = 2
        gap = next_gap

    spans = np.full((N, 2), -1, dtype=np.int64)
    t, i = T, N
    in_token = False
    while t > 0 or i > 0 or in_token:
        if in_token:
            if spans[i, 1] < 0:
                spans[i, 1] = t
            spans[i, 0] = t - 1
            entered = token_back[t, i]
            t -= 1
            if entered == 1:
                in_token = False
        else:
            source = gap_back[t, i]
            if source == 0:
                t -= 1
            elif source == 1:
                i -= 1
                in_token = True
            else:
                i -= 1
                spans[i, 0] = t
                spans[i, 1] = t
    _canonicalize_skipped(spans, T, gap_allowed)
    return spans


def _decode_flat(sim: np.ndarray, groups: np.ndarray | None,
                 skip_penalty: float = 0.5) -> np.ndarray:
    """sim [T,N]，groups [N]（同组 token 间禁止等待）。返回 [N,2] 帧区间。"""
    T, N = sim.shape
    gap_allowed = np.ones(N + 1, dtype=np.bool_)
    if groups is not None:
        for i in range(1, N):
            gap_allowed[i] = groups[i - 1] != groups[i]
    return _decode_flat_single(sim.astype(np.float32), skip_penalty, gap_allowed)


# ---------------------------------------------------------------------------
# 编排：音频 + PFML → 词/音素区间
# ---------------------------------------------------------------------------

def _load_wave_48k(wav_path, sr_target: int):
    data, s0 = sf.read(str(wav_path), always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if s0 != sr_target:
        from math import gcd
        from scipy import signal
        g = gcd(s0, sr_target)
        data = signal.resample_poly(data, sr_target // g, s0 // g).astype(np.float32)
    return data


# ---------------------------------------------------------------------------
# 对齐质量统计（reference-free metrics：confidence / determinacy / monotonicity）
# ---------------------------------------------------------------------------

def _compute_alignment_metrics(sim: np.ndarray, spans: np.ndarray, maskT: np.ndarray,
                               maskN: np.ndarray) -> dict:
    """计算单条样本的对齐质量指标（纯 numpy，无 torch）。

    Args:
        sim: [T, N] similarity 矩阵
        spans: [N, 2] Viterbi spans（帧）
        maskT: [T] bool，有效帧
        maskN: [N] bool，有效 token
    Returns:
        {confidence, determinacy, monotonicity, num_skipped_tokens, num_frames}
    """
    T, N = sim.shape
    # confidence: mean similarity within spans
    t_idx = np.arange(T).reshape(T, 1)
    onsets = spans[:, 0].reshape(1, N)
    offsets = spans[:, 1].reshape(1, N)
    span_mask = (t_idx >= onsets) & (t_idx < offsets)
    span_mask = span_mask & maskT.reshape(T, 1) & maskN.reshape(1, N)
    token_sum = (sim * span_mask).sum(axis=0)
    token_count = span_mask.sum(axis=0)
    token_conf = np.zeros(N, dtype=np.float32)
    nz = token_count > 0
    token_conf[nz] = token_sum[nz] / token_count[nz]
    valid_count = maskN.sum()
    confidence = float(token_conf.sum() / max(valid_count, 1))

    # determinacy: sum(a_assigned) / sum(a_local), a = relu(sim)**2, width=5
    a = np.maximum(0.0, sim) ** 2.0
    a[~maskT.reshape(T, 1) | ~maskN.reshape(1, N)] = 0.0
    width = 5
    # local sum over token dim with width=5
    local_sum = np.zeros_like(a)
    for j in range(N):
        lo, hi = max(0, j - width), min(N, j + width + 1)
        local_sum[:, j] = a[:, lo:hi].sum(axis=1)

    # frame assignment (same as monotonicity logic)
    onset_f = spans[:, 0].astype(np.float32).copy()
    onset_f[~maskN] = np.inf
    offset_f = spans[:, 1].astype(np.float32)
    pos = np.searchsorted(onset_f, np.arange(T), side="right") - 1
    pos_clamped = np.clip(pos, 0, N - 1)
    is_token = (pos >= 0) & (np.arange(T) < offset_f[pos_clamped])
    regions = np.where(is_token, pos_clamped + 1, 0)
    N_valid = max(int(maskN.sum()) - 1, 0)
    gap_pos = np.clip(pos + 1, 0, N_valid)
    denom_idx = np.where(is_token, pos_clamped, gap_pos)
    a_padded = np.pad(a, ((0, 0), (1, 0)), constant_values=0.0)
    num_contrib = a_padded[np.arange(T), regions]
    num_contrib[~maskT] = 0.0
    denom_contrib = local_sum[np.arange(T), denom_idx]
    denom_contrib[~maskT] = 0.0
    determinacy = float(num_contrib.sum() / max(denom_contrib.sum(), 1e-8))

    # monotonicity: forward_mass / total_mass
    cumsum_rev = np.cumsum(a[:, ::-1], axis=1)[:, ::-1]
    forward_mass = cumsum_rev[np.arange(T), pos_clamped]
    total_mass = a.sum(axis=1)
    backward_mass = total_mass - forward_mass
    total_fb = forward_mass + backward_mass
    valid_m = is_token & maskT & (total_fb > 0)
    per_frame = forward_mass / np.maximum(total_fb, 1e-8)
    per_frame[~valid_m] = 0.0
    monotonicity = float(per_frame.sum() / max(valid_m.sum(), 1))

    # skipped tokens (zero-width spans)
    num_skipped = int((spans[:, 0] >= spans[:, 1]).sum())

    return {
        "confidence": round(confidence, 6),
        "determinacy": round(determinacy, 6),
        "monotonicity": round(monotonicity, 6),
        "num_skipped_tokens": num_skipped,
        "num_frames": int(maskT.sum()),
    }


def tifa_align_audio(wav_path, pfml_text: str, lang: str,
                     extra_langs: list[str] | None = None,
                     main_lang: str | None = None) -> dict:
    """TIFA ONNX 对齐。返回 {words, prons, phones, duration, metrics}（秒）。

    extra_langs: 副语言列表（对应 TIFA -L），激活对应 G2P 以便副语言音素正确解析。
    main_lang: 主语言（对应 TIFA -l），输出标签剥除其前缀；默认取 lang。
    """
    import g2p  # LS 的 g2pflow 管道（懒加载 MeCab/词典）
    cfg = _tifa_config()
    sr, timestep = cfg["samplerate"], cfg["timestep"]

    wave = _load_wave_48k(wav_path, sr)
    duration = len(wave) / sr
    min_len = cfg["win_size"] + cfg["hop_size"]
    if len(wave) < min_len:
        wave = np.pad(wave, (0, min_len - len(wave)))
    spec, maskT = _tifa_session("spectrogram").run(None, {
        "waveform": wave[None, :],
        "duration": np.array([duration], dtype=np.float32),
    })

    languages = [lang] + list(extra_langs or [])
    g2p_words = g2p._g2pflow_pipeline().convert_pfml(pfml_text, languages=languages)
    vocab = _tifa_vocabulary()
    data, lexicon, texts = _encode_paths(g2p_words, vocab, languages)
    if not data["paths"].any():
        return {"words": [], "prons": [], "phones": [],
                "duration": duration, "metrics": {}}
    paths = data["paths"][None]
    words_ = data["words"][None]
    groups = data["groups"][None]
    candidates = data["candidates"][None]

    tokens, segments, mapping = _tifa_session("prepare").run(None, {
        "paths": paths, "words": words_, "candidates": candidates,
        "grouped": np.array(False, dtype=np.bool_),
    })
    choices = candidates.any(axis=-1).astype(np.int64)  # [1,W]，候选前缀打包 → 1=首选
    word_scores = np.full(candidates.shape[1:], -np.inf, dtype=np.float64)  # [W,C]

    if (segments > 0).any():
        # 有歧义段：模型打分 + 整词 DP
        # logits 是未归一化输出；score.onnx 内部自行 log_softmax，传 raw logits
        logits = _tifa_session("model").run(None, {
            "spectrogram": spec, "tokens": tokens,
            "maskT": maskT, "maskN": tokens != 0,
        })[1]
        descriptors, lengths, costs, tails, capacity = _tifa_session("score").run(None, {
            "logits": logits, "paths": paths, "words": words_,
            "segments": segments, "mapping": mapping,
        })
        try:
            choices[0], word_scores = _select_choices(candidates[0], descriptors[0],
                                                      lengths[0], costs[0], tails[0],
                                                      capacity[0])
            # 与 TIFA 一致：按 MASK 容量归一 + log(vocab_size)
            count = int(capacity[0].sum())
            if count:
                word_scores = word_scores / count + math.log(cfg["vocab_size"])
        except ValueError:
            pass  # 无可行路径 → 保留首选

    best_tokens, best_words, best_groups, maskN = _tifa_session("select").run(None, {
        "paths": paths, "words": words_, "groups": groups, "choices": choices,
    })
    N = int(maskN[0].sum())
    T = int(maskT[0].sum())
    if N == 0 or T == 0:
        return {"words": [], "prons": [], "phones": [],
                "duration": duration, "metrics": {}}

    sim, tok_logits = _tifa_session("model").run(None, {
        "spectrogram": spec, "tokens": best_tokens,
        "maskT": maskT, "maskN": maskN,
    })
    spans = _decode_flat(sim[0, :T, :N], best_groups[0, :N])  # [N,2] 帧

    # 对齐质量统计
    metrics = _compute_alignment_metrics(
        sim[0, :T, :N], spans, maskT[0, :T], maskN[0, :N],
    )
    metrics["num_tokens"] = N
    # agreement = 选中 token 的平均 softmax 概率
    tl = tok_logits[0, :N].astype(np.float64)
    tl = tl - tl.max(axis=-1, keepdims=True)
    probs = np.exp(tl)
    probs /= probs.sum(axis=-1, keepdims=True)
    tok_probs = probs[np.arange(N), best_tokens[0, :N]]
    metrics["agreement"] = round(float(tok_probs[maskN[0, :N]].mean()), 6)

    # 词 → 选中候选的音素串（与 list_candidates 可比）
    word_phones = {}
    for w in range(len(texts)):
        c = int(choices[0, w])
        if c > 0 and c - 1 < len(lexicon[w]):
            word_phones[w + 1] = lexicon[w][c - 1]["phonemes"]

    # 输出时剥掉主语言前缀（与 TIFA -l 一致）：主语言 zh/d → d，副语言 en/d 保持 en/d
    _main_prefix = f"{main_lang or lang}/"
    def _strip_main(s: str) -> str:
        return s[len(_main_prefix):] if s.startswith(_main_prefix) else s

    # 展平的 group_scripts：按词序拼接选中候选的 scripts（对齐 TIFA group_scripts）
    group_scripts = []
    for w in range(len(texts)):
        c = int(choices[0, w])
        if c > 0 and c - 1 < len(lexicon[w]):
            group_scripts.extend(lexicon[w][c - 1]["scripts"])

    # 逐 token 保留列表：(t0, t1, 词ID, 全局groupID, 音素label)，零宽 token 丢弃
    tok = best_tokens[0, :N]
    wids = best_words[0, :N]
    gids = best_groups[0, :N]
    kept = []
    counters: dict[int, int] = {}
    for i in range(N):
        w = int(wids[i])
        k = counters.get(w, 0)
        counters[w] = k + 1
        phs = word_phones.get(w) or []
        label = phs[k] if k < len(phs) else vocab.label(tok[i])
        label = _strip_main(label)
        t0, t1 = int(spans[i, 0]), int(spans[i, 1])
        if label and t1 > t0:
            kept.append((t0 * timestep, t1 * timestep, w, int(gids[i]), label))

    phones = [(t0, t1, lab) for t0, t1, _w, _g, lab in kept]

    def _aggregate(key_idx: int, label_of) -> list:
        """按 owner（词ID/组ID）聚合连续 kept token → (t0, t1, label) 区间。"""
        items = []
        i = 0
        while i < len(kept):
            owner = kept[i][key_idx]
            j = i
            while j + 1 < len(kept) and kept[j + 1][key_idx] == owner:
                j += 1
            items.append((kept[i][0], kept[j][1], label_of(owner)))
            i = j + 1
        return items

    # texts 层：按语义词聚合，label=表面字符
    word_items = _aggregate(2, lambda w: texts[w - 1] if 0 < w <= len(texts) else "")
    # words 层：按发音组聚合，label=G2P script 读音（对齐 TIFA，如 我→wo、是→shi）
    pron_items = _aggregate(
        3, lambda g: group_scripts[g - 1] if 0 < g <= len(group_scripts) else "")

    # scores.json 记录：多候选词的选音与备选分数（与 TIFA StatisticsCallback 同构）
    score_words = []
    for w in range(len(texts)):
        cands = lexicon[w]
        if len(cands) <= 1:
            continue
        score_words.append({
            "index": w,
            "text": texts[w],
            "chosen": int(choices[0, w]) - 1,
            "alternatives": [
                {"index": c,
                 "scripts": cands[c].get("scripts", []),
                 "phones": [_strip_main(p) for p in cands[c]["phonemes"]],
                 "score": float(word_scores[w, c])}
                for c in range(len(cands))
            ],
        })
    return {"words": word_items, "prons": pron_items, "phones": phones,
            "duration": duration, "metrics": metrics, "score_words": score_words}


def _write_textgrid(path, words, prons, phones, duration: float):
    """写 ooTextFile TextGrid（texts/words/phones 三层），与 cpp 输出同构。"""
    def esc(s: str) -> str:
        return s.replace('"', '""')

    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
             "xmin = 0", f"xmax = {duration:.6f}", "tiers? <exists>", "size = 3", "item []:"]
    for idx, (name, items) in enumerate(
            (("texts", words), ("words", prons), ("phones", phones)), 1):
        lines += [f"    item [{idx}]:", '        class = "IntervalTier"',
                  f'        name = "{name}"', "        xmin = 0",
                  f"        xmax = {duration:.6f}", f"        intervals: size = {len(items)}"]
        for i, (t0, t1, lab) in enumerate(items, 1):
            lines += [f"        intervals [{i}]:", f"            xmin = {t0:.6f}",
                      f"            xmax = {t1:.6f}", f'            text = "{esc(lab)}"']
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_align_onnx(wav_path, pfml_text: str, lang: str, out_dir,
                     extra_langs: list[str] | None = None,
                     main_lang: str | None = None) -> tuple[bool, Path | None, str]:
    """ONNX 版对齐入口。输出 texts/words/phones 三层 TextGrid。

    extra_langs: 副语言列表（对应 TIFA -L），激活对应 G2P。
    main_lang: 主语言（对应 TIFA -l），输出标签剥除其前缀。
    返回 (ok, textgrid_path, error, metrics)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tg = out_dir / (Path(wav_path).stem + ".TextGrid")
    try:
        result = tifa_align_audio(wav_path, pfml_text, lang, extra_langs,
                                  main_lang=main_lang)
    except Exception as e:
        return False, None, f"ONNX 对齐失败: {e}", {}
    _write_textgrid(tg, result["words"], result["prons"], result["phones"],
                    result["duration"])
    extra = {"metrics": result.get("metrics", {}),
             "score_words": result.get("score_words", [])}
    return True, tg, "", extra


# ---------------------------------------------------------------------------
# 对齐质量统计导出（对齐 TIFA StatisticsCallback：diagnosis.json + scores.json + 图表）
# ---------------------------------------------------------------------------

_STAT_KEYS = ["confidence", "determinacy", "monotonicity", "agreement"]


def _metric_histogram(values: np.ndarray, label: str):
    """单指标直方图（mean/median/百分位线），对齐 TIFA metric_histogram_figure。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=80, alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.set_title(label, fontsize=15)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    mv, md = float(values.mean()), float(np.median(values))
    ax.axvline(mv, color="red", linestyle="--", linewidth=1, label=f"mean={mv:.4f}")
    ax.axvline(md, color="orange", linestyle=":", linewidth=1, label=f"median={md:.4f}")
    for p, c in ((1, "mediumseagreen"), (5, "seagreen"), (10, "green")):
        v = float(np.percentile(values, p))
        ax.axvline(v, color=c, linestyle="-.", linewidth=1,
                   label=f"keep{100 - p}% >= {v:.4f}")
    ax.legend()
    fig.tight_layout()
    return fig


def _metric_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str):
    """指标 vs num_frames 散点图，对齐 TIFA metric_scatter_figure。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, alpha=0.8, s=4, edgecolors="none")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.axhline(float(y.mean()), color="red", linestyle="--", linewidth=0.8,
               alpha=0.7, label="mean")
    ax.axvline(float(x.mean()), color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    return fig


def write_statistics(stat_dir, records: list[dict], score_records: list[dict]):
    """写 statistics：diagnosis.json（按 skipped tokens 排序）+ scores.json + 指标图表。

    records: [{identifier, confidence, determinacy, monotonicity, agreement,
               num_frames, num_tokens, num_skipped_tokens}, ...]
    score_records: [{identifier, words: [{index, text, chosen, alternatives}]}]
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stat_dir = Path(stat_dir)
    stat_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    # 排序：num_skipped_tokens 降序，然后按各指标排名（对齐 TIFA _sort_key 简化版）
    available = [k for k in _STAT_KEYS if k in records[0]]
    ranks = {}
    for key in available:
        ordered = sorted(records, key=lambda r: r[key])  # 低=差
        ranks[key] = {r["identifier"]: i for i, r in enumerate(ordered)}

    def _sort_key(r):
        ident = r["identifier"]
        rks = tuple(ranks[k][ident] for k in available)
        return (-r.get("num_skipped_tokens", 0), min(rks)) + rks

    records.sort(key=_sort_key)
    (stat_dir / "diagnosis.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (stat_dir / "scores.json").write_text(
        json.dumps(score_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # 图表：histogram + scatter(vs num_frames)
    has_frames = "num_frames" in records[0]
    frames = np.array([r["num_frames"] for r in records], dtype=np.int32) if has_frames else None
    for key in available:
        values = np.array([r[key] for r in records], dtype=np.float32)
        fig = _metric_histogram(values, label=key)
        fig.savefig(stat_dir / f"{key}_histogram.jpg", bbox_inches="tight")
        plt.close(fig)
        if has_frames:
            fig = _metric_scatter(frames, values, "Number of Frames", key, key)
            fig.savefig(stat_dir / f"{key}_scatter.jpg", bbox_inches="tight")
            plt.close(fig)
