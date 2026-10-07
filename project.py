# coding=utf-8
"""工程文件管理：.labproj（JSON）。

结构（用户已确认）：
{
  "folder_path": "D:/audio_dataset",
  "wavs": {
    "song01.wav": [
      {"start": 0.250, "end": 4.120, "text": "歌词", "lang": "zh"}
    ]
  }
}
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from char_id import CharIdMap
from slots import SlotMap, get_alloc, set_alloc_next

PROJECT_EXT = ".labproj"


def seconds(v: float) -> float:
    """统一保留 3 位小数（用户确认：时间用秒，保留 3 位）。"""
    return round(float(v), 3)


def _pfml_is_uid(pfml: dict) -> bool:
    """判断 pfml 是否 uid 版（含 su/eu）。位置版含 begin/end。"""
    for k in ("words", "overrides", "spans"):
        for it in (pfml or {}).get(k, []):
            return "su" in it or "eu" in it
    return False


class Mark:
    """单个标记片段。

    采用全局唯一 uid 的"格子"模型（新构筑）：
    - self.slots: SlotMap，维护 text 的格子(uid)划分（zh/yue 逐字、en 逐词、ja 分词器分组）
    - self.pfml: uid 版标注 {words:[{su,eu}], overrides:[{su,eu,...}], spans:[{su,eu,lang}]}
    - 渲染/导出前用 slots.to_pos_pfml() 转成位置版
    文本变更：set_text 用方案B diff 锚定；match 用 apply_matched_text 整段重切分（日语自动分词）
    """

    __slots__ = ("start", "end", "text", "lang", "pfml", "slots", "_match_changed")

    def __init__(self, start: float, end: float, text: str = "", lang: str = "zh",
                 pfml=None, slots: SlotMap | None = None):
        self.start = seconds(start)
        self.end = seconds(end)
        self.text = text
        self.lang = lang
        if slots is not None:
            self.slots = slots
            self.pfml = pfml or {}
        else:
            self.slots = SlotMap(get_alloc(), text, lang)
            if pfml:
                if _pfml_is_uid(pfml):
                    self.pfml = pfml  # 调用方应传 slots（复制场景请用 copy()）
                else:
                    self.pfml = self.slots.to_uid_pfml(pfml)  # 位置版→uid版
            else:
                self.pfml = {}
        self._match_changed = False  # 运行时用：match 歌词后是否改动过（不存盘）

    @property
    def duration(self) -> float:
        return seconds(max(0.0, self.end - self.start))

    def set_text(self, new_text: str):
        """编辑文本并方案B同步格子 uid（标注自动跟随，不现算编号）。"""
        if new_text == self.text:
            return
        self.slots.sync(new_text, self.lang)
        self.text = new_text

    def apply_matched_text(self, new_text: str):
        """match 整段替换：重切分、全部发新 uid；日语自动分词；旧标注作废。"""
        self.slots.reset(new_text, self.lang)
        self.text = new_text
        self.pfml = {}

    def pos_pfml(self) -> dict:
        """返回位置版 pfml（begin/end），供 pfml_strip 渲染 / to_pfml 导出。"""
        if not self.pfml:
            return {"words": [], "overrides": [], "spans": []}
        return self.slots.to_pos_pfml(self.pfml)

    def cancel_overrides_for(self, uid_list: list[int]):
        """取消绑定在指定 uid 方格上的发音与插入音素（防错：内容变了的方格）。"""
        if not uid_list:
            return
        changed = set(uid_list)
        keep = []
        for o in self.pfml.get("overrides", []):
            try:
                su, eu = int(o["su"]), o.get("eu")
            except Exception:
                continue
            if su == eu:
                hit = su in changed
            else:
                hit = any(su <= u < eu for u in changed)
            if hit:
                continue  # 取消
            keep.append(o)
        if len(keep) != len(self.pfml.get("overrides", [])):
            self.pfml["overrides"] = keep

    def _prune_touching(self, uid_list: list[int], keep_spans: bool = False):
        """删除 pfml 中与这些 uid 相交的标注（合并/拆分后旧格失效）。

        keep_spans=True：语言标签(spans)保留（取消分词不应顺带删掉语言设置），
        调用方需先自行把 span 端点从旧 uid 重锚到新 uid。"""
        uids = set(uid_list)
        if not uids:
            return

        def overlap(it) -> bool:
            try:
                su = int(it.get("su"))
            except Exception:
                return True
            eu = it.get("eu")
            if eu is None:
                # 到末尾：任何 >su 的旧格被包含
                return any(u >= su for u in uids)
            try:
                eu = int(eu)
            except Exception:
                return True
            if eu == su:
                # 插入音素/点标注：绑定在 su 格上，su 是旧格即删
                return su in uids
            return any(su <= u < eu for u in uids)

        out = {}
        for k in ("words", "overrides", "spans"):
            if keep_spans and k == "spans":
                out[k] = self.pfml.get(k, [])
            else:
                out[k] = [it for it in self.pfml.get(k, []) if not overlap(it)]
        self.pfml = out

    def merge_slots(self, b: int, e: int) -> bool:
        """日语手动合并 [b,e) 内相邻格子为一格；合并后清空该范围旧标注。

        规则：UID 失效即连带删除其所有标记（词边界/发音/插入音素），不重锚。"""
        old = {self.slots.uid_at(k) for k in range(b, e) if self.slots.uid_at(k) >= 0}
        u = self.slots.merge(b, e)
        if u is None:
            return False
        self._prune_touching(old)
        return True

    def ungroup_slot(self, pos: int) -> bool:
        """取消分词：把 pos 所在词格拆回逐字格。

        旧格上的词边界/发音/插入音素作废；**语言标签(spans)保留**——把引用旧格
        uid 的 span 端点重锚到拆出的新逐字格，避免中键取消分词时误删语言设置。"""
        orig = self.slots.uid_at(pos) if 0 <= pos < len(self.slots.uids) else -1
        if orig < 0:
            orig = self.slots.uid_at(pos - 1) if pos > 0 else -1
        ok = self.slots.ungroup(pos)
        if not ok or orig < 0:
            return ok
        new_uids = list(self.slots.last_changed)
        if new_uids:
            first_u, last_u = new_uids[0], new_uids[-1]
            for s in self.pfml.get("spans", []):
                try:
                    if int(s.get("su")) == orig:
                        s["su"] = first_u
                    eu = s.get("eu")
                    if eu is not None and int(eu) == orig:
                        s["eu"] = last_u
                except Exception:
                    continue
        self._prune_touching([orig], keep_spans=True)
        return True

    def copy(self) -> "Mark":
        import copy
        return Mark(self.start, self.end, self.text, self.lang,
                    copy.deepcopy(self.pfml), self.slots.clone())

    def to_dict(self) -> dict:
        d = {"start": self.start, "end": self.end, "text": self.text, "lang": self.lang}
        if self.pfml:
            d["pfml"] = self.pfml
        d["slots"] = self.slots.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mark":
        text = str(d.get("text", ""))
        lang = str(d.get("lang", "zh") or "zh")
        pfml_raw = d.get("pfml", {}) or {}
        slots_raw = d.get("slots")
        charid_raw = d.get("charid")
        if slots_raw:
            slots = SlotMap.from_dict(slots_raw, get_alloc())
            # 旧工程可能残留位置版 pfml；强制转 uid 版避免 to_pos_pfml 异常
            if pfml_raw and not _pfml_is_uid(pfml_raw):
                pfml = slots.to_uid_pfml(pfml_raw)
            else:
                pfml = pfml_raw or {}
        elif charid_raw:
            # 旧工程迁移：局部 uid → 全局 uid，pfml 按位置重锚
            old_charid = CharIdMap.from_dict(text, charid_raw)
            pos_pfml = old_charid.to_pos_pfml(pfml_raw) if pfml_raw else {}
            slots = SlotMap(get_alloc(), text, lang)
            pfml = slots.to_uid_pfml(pos_pfml) if pos_pfml else {}
        else:
            slots = SlotMap(get_alloc(), text, lang)
            pfml = slots.to_uid_pfml(pfml_raw) if pfml_raw else {}
        return cls(
            float(d.get("start", 0)),
            float(d.get("end", 0)),
            text,
            lang,
            pfml,
            slots,
        )


class Project:
    """一个工程：对应一个工作文件夹下的多个 wav。"""

    def __init__(self):
        self.folder_path: str = ""
        self.wavs: dict[str, list[Mark]] = {}
        self.lyrics: dict[str, str] = {}  # 每个 wav 的歌词原文，按文件名存
        self.project_file: Path | None = None
        self.dirty = False  # 是否有未保存修改
        self.last_saved_mtime = time.time()

    @property
    def wav_names(self) -> list[str]:
        """按文件夹内顺序返回 wav 文件名。"""
        # 尽量按磁盘实际文件顺序
        if self.folder_path:
            try:
                disk = [p.name for p in sorted(Path(self.folder_path).glob("*.wav"))]
            except Exception:
                disk = []
            merged = []
            seen = set()
            for n in disk + list(self.wavs.keys()):
                if n not in seen:
                    seen.add(n)
                    merged.append(n)
            return merged
        return list(self.wavs.keys())

    def marks_of(self, wav_name: str) -> list[Mark]:
        if wav_name not in self.wavs:
            self.wavs[wav_name] = []
        return self.wavs[wav_name]

    def set_marks(self, wav_name: str, marks: list[Mark]):
        self.wavs[wav_name] = marks
        self.dirty = True

    def add_mark(self, wav_name: str, mark: Mark):
        marks = self.marks_of(wav_name)
        marks.append(mark)
        marks.sort(key=lambda m: m.start)
        self.dirty = True

    def remove_marks(self, wav_name: str, indices: list[int]):
        marks = self.marks_of(wav_name)
        if not marks:
            return
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(marks):
                del marks[idx]
        self.dirty = True

    # ---- 歌词 ----
    def lyric_of(self, wav_name: str) -> str:
        return self.lyrics.get(wav_name, "")

    def set_lyric(self, wav_name: str, text: str):
        cur = self.lyrics.get(wav_name, "")
        if text == cur:
            return
        self.lyrics[wav_name] = text
        self.dirty = True

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        d = {
            "folder_path": self.folder_path,
            "wavs": {
                name: [m.to_dict() for m in marks]
                for name, marks in sorted(self.wavs.items())
            },
            "uid_alloc": get_alloc().to_dict(),  # 全局 uid 分配器推进值
        }
        if self.lyrics:
            d["lyrics"] = dict(sorted(self.lyrics.items()))
        return d

    def save(self, path: Path | None = None):
        target = Path(path) if path else self.project_file
        if target is None:
            raise ValueError("未指定工程文件路径")
        target = target if target.suffix == PROJECT_EXT else target.with_suffix(PROJECT_EXT)
        payload = self.to_dict()
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.project_file = target
        self.dirty = False
        self.last_saved_mtime = time.time()
        return target

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        p = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 恢复全局 uid 分配器（旧工程无此字段则从 0 起重新分配，并迁移）
        alloc_d = data.get("uid_alloc") or {}
        set_alloc_next(int(alloc_d.get("next", 0)))
        p.folder_path = str(data.get("folder_path", ""))
        p.wavs = {}
        for name, marks in (data.get("wavs") or {}).items():
            p.wavs[str(name)] = [Mark.from_dict(m) for m in (marks or [])]
        p.lyrics = {}
        for name, text in (data.get("lyrics") or {}).items():
            p.lyrics[str(name)] = str(text)
        p.project_file = Path(path)
        p.dirty = False
        p.last_saved_mtime = time.time()
        return p

    # ---- 自动保存辅助 ----
    def make_backup(self, path: Path | None = None):
        """自动保存前保留一份 .bak 备份（用户确认：可保留备份防损坏）。"""
        target = Path(path) if path else self.project_file
        if target is None or not target.is_file():
            return
        try:
            bak = target.with_suffix(".labproj.bak")
            shutil.copyfile(target, bak)
        except Exception:
            pass
