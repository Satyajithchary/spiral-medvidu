"""
Torch Dataset + collator. Turns a harvested row into Qwen3-VL processor inputs.

Design notes:
  - We pass frames as a *video* (not N separate images) so the model's temporal
    position encoding is used, and we additionally state the timestamps in text.
    Video path = ~half the visual tokens of the image path and keeps MRoPE
    temporal structure. The text grid supplies the exact times.
  - The anchor frame for region-caption tasks is ALSO appended as a separate
    high-resolution image with the box drawn on it. Visual prompting beats
    textual box coordinates for small objects, and the MedGRPO authors' own
    ablation says their pipeline relies on box overlays.
  - Labels mask everything but the assistant turn.
"""
from __future__ import annotations
import json, os, torch
from typing import Any
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .sampling import select_frames, anchor_index, normalize_task
from .prompts import build_system, build_user, get_answer

IGNORE = -100


def remap_path(p: str, src_prefix: str, cache_root: str) -> str:
    """src_prefix may be '' — the test split stores relative paths."""
    if not cache_root:
        return p
    if src_prefix and p.startswith(src_prefix):
        return os.path.join(cache_root, p[len(src_prefix):].lstrip("/"))
    if not src_prefix:
        return os.path.join(cache_root, p.lstrip("/"))
    return p


def draw_box(path: str, box, out_size=448) -> Image.Image:
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(2, int(0.006 * max(im.size)))
    d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=w)
    if max(im.size) > out_size:
        s = out_size / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
    return im


class MedVidDataset(Dataset):
    def __init__(self, json_path, processor, src_prefix="/root/data",
                 cache_root=None, budget_scale=1.0, max_frames=None,
                 max_len=8192, train=True, box_overlay=True):
        with open(json_path) as f:
            self.rows = json.load(f)
        self.pr = processor
        self.src_prefix = src_prefix
        self.cache_root = cache_root
        if not cache_root:
            print("\n" + "!" * 70)
            print("[dataset] cache_root is EMPTY. Every frame will be read from the")
            print("[dataset] original paths in the json. If that prefix is absent or")
            print("[dataset] unreadable, every row fails and the run still reports")
            print("[dataset] completion. Pass --cache explicitly, or run")
            print("[dataset]   python -m src.paths --root <root> --write")
            print("!" * 70 + "\n")
        self.budget_scale = budget_scale
        self.max_frames = max_frames
        self.max_len = max_len
        self.train = train
        self.box_overlay = box_overlay

    def __len__(self):
        return len(self.rows)

    # ---------------------------------------------------------------- content
    def build_messages(self, row, ctcd: str | None = None,
                       force_idx: list[int] | None = None):
        """force_idx overrides frame selection — used by C4 temporal zoom to
        re-sample densely inside the model's own coarse hypothesis window."""
        if force_idx is not None:
            from .sampling import frame_times as _ft
            t_all = _ft(row)
            idx = [i for i in force_idx if 0 <= i < len(row["video"])]
            times = [float(t_all[i]) for i in idx]
            mode = "zoom"
        else:
            idx, times, mode = select_frames(row, self.cache_root,
                                             self.budget_scale, self.max_frames)
        frames = [remap_path(row["video"][i], self.src_prefix, self.cache_root)
                  for i in idx]
        frames = [p for p in frames if os.path.exists(p)]
        if not frames:
            # Fall back to the original paths only when they are actually
            # readable. On a machine where the original prefix exists but is
            # root owned, os.path.exists succeeds and the later read raises
            # PermissionError, which aborts the row for no useful reason.
            raw = [row["video"][i] for i in idx]
            frames = [p for p in raw if os.access(p, os.R_OK)]
            if not frames:
                raise FileNotFoundError(
                    f"no readable frame for {row.get('id')} "
                    f"({row.get('dataset_name')}). Cache miss at "
                    f"{remap_path(raw[0], self.src_prefix, self.cache_root)} "
                    f"and original unreadable at {raw[0]}. "
                    f"Run: python -m src.check_cache --gt <split> --fix")

        content: list[dict[str, Any]] = [{"type": "video", "video": frames}]

        # visual prompt: anchor frame with the box burned in
        rc = row.get("RC_info") or {}
        if self.box_overlay and rc.get("start_frame_bbox"):
            a = anchor_index(row)
            if a is not None:
                ap = remap_path(row["video"][a], self.src_prefix, self.cache_root)
                if os.path.exists(ap) and os.access(ap, os.R_OK):
                    try:
                        content.append({"type": "image",
                                        "image": draw_box(ap, rc["start_frame_bbox"])})
                        content.append({"type": "text",
                                        "text": "The image above is the anchor "
                                                "frame with the target region "
                                                "outlined in red."})
                    except Exception:
                        pass

        content.append({"type": "text", "text": build_user(row, times)})
        # C3: during training the row may carry a pre-computed noised prior
        if ctcd is None:
            ctcd = row.get("_ctcd_prior")
        msgs = [{"role": "system",
                 "content": [{"type": "text", "text": build_system(row, ctcd)}]},
                {"role": "user", "content": content}]
        return msgs, times

    # ------------------------------------------------------------------ item
    def __getitem__(self, i):
        row = self.rows[i]
        msgs, _ = self.build_messages(row)
        answer = get_answer(row) or ""
        return {"messages": msgs, "answer": answer, "row": row}


def make_collator(processor, max_len=8192, train=True):
    tok = processor.tokenizer

    def collate(batch):
        texts, videos_in, images_in = [], [], []
        for b in batch:
            msgs = b["messages"]
            if train:
                msgs = msgs + [{"role": "assistant",
                                "content": [{"type": "text", "text": b["answer"]}]}]
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=not train))

        # qwen_vl_utils handles both qwen2.5-vl and qwen3-vl message dicts
        from qwen_vl_utils import process_vision_info
        for b in batch:
            imgs, vids = process_vision_info(b["messages"])
            images_in.append(imgs)
            videos_in.append(vids)

        flat_i = [x for lst in images_in if lst for x in lst] or None
        flat_v = [x for lst in videos_in if lst for x in lst] or None

        enc = processor(text=texts, images=flat_i, videos=flat_v,
                        padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt")
        if not train:
            enc["_rows"] = [b["row"] for b in batch]
            return enc

        labels = enc["input_ids"].clone()
        labels[labels == tok.pad_token_id] = IGNORE
        # mask everything before the assistant turn
        start_ids = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        L = len(start_ids)
        ids = enc["input_ids"]
        for r in range(ids.size(0)):
            seq = ids[r].tolist()
            pos = -1
            for j in range(len(seq) - L, -1, -1):
                if seq[j:j + L] == start_ids:
                    pos = j + L
                    break
            if pos > 0:
                labels[r, :pos] = IGNORE
            else:                       # couldn't find it -> drop the row
                labels[r, :] = IGNORE
        # never train on visual placeholder tokens
        for name in ("image_token_id", "video_token_id"):
            tid = getattr(processor, name, None) or \
                getattr(getattr(processor, "tokenizer", None), name, None)
            if isinstance(tid, int):
                labels[ids == tid] = IGNORE
        enc["labels"] = labels
        return enc

    return collate
