"""
Prompt construction.

Three things happen here that the baselines don't do:

1. TIMESTAMP GRID. Every prompt states the exact second each supplied frame was
   taken from. This is what makes adaptive/non-uniform sampling safe, and it is
   almost certainly a large part of why Qwen3-VL (timestamp-aware) crushes
   Qwen2.5-VL on TAG in the published table.

2. DATASET DIALECT. The LLM judge scores *similarity to the reference*, and the
   reference was written in the source dataset's vocabulary. Telling the model
   "this is CholecT50, instruments are exactly {grasper, bipolar, hook,
   scissors, clipper, irrigator}" is worth free points on R1/R2.

3. CTCD CONTEXT SLOT. At inference, captioning prompts for a video can be given
   the model's OWN previously-predicted action timeline for that same video.
   No ground truth, no leakage — pure test-time compute. See infer.py.
"""
from __future__ import annotations
from .ontology import dataset_hint
from .sampling import normalize_task

BASE_ROLE = (
    "You are an expert surgical video analyst with board-level knowledge of "
    "operative anatomy, instrumentation, and procedural workflow."
)

OUTPUT_SPEC = {
    "tal":
        "Answer ONLY with the time segments, comma separated on one line, "
        "ending with the word seconds and a full stop. Exactly like this:\n"
        "20.7-55.7 seconds.\n"
        "or, for several occurrences:\n"
        "66.0-66.0, 124.0-126.0, 144.0-154.0 seconds.\n"
        "One decimal place. Instantaneous events are written with an identical "
        "start and end (66.0-66.0) and must still be reported. List every "
        "occurrence. If the action never occurs, answer exactly: none",
    "stg":
        "Answer ONLY with a bounding box for EACH requested timestamp, in the "
        "exact format:\n"
        "66.0 seconds: [117.54, 339.56, 626.88, 1005.62] "
        "74.0 seconds: [104.48, 313.44, 587.70, 1005.62]\n"
        "Pixel coordinates [x1, y1, x2, y2] in the ORIGINAL frame resolution, "
        "top-left origin, two decimals. Give one box per timestamp asked about "
        "— a single box is not a valid answer. No other text.",
    "dense_captioning":
        "Answer with one line per qualifying segment, in the exact format:\n"
        "<start>-<end> seconds: <action name>: <one-sentence description>\n"
        "If the source provides no action vocabulary, omit the action field:\n"
        "<start>-<end> seconds: <one-sentence description>\n"
        "Name the action using the exact wording from the provided action list. "
        "Describe only what is visually observable. Do not include segments for "
        "actions outside the list.",
    "video_summary":
        "Answer with a single concise paragraph describing what happens over "
        "time. Name specific instruments and anatomical structures. Emphasise "
        "motion, interaction and anatomical change. No preamble, no bullet points.",
    "region_caption":
        "Answer with ONE sentence describing the activity or state of the object "
        "inside the bounding box over the given interval. Name the instrument "
        "specifically, name the anatomy specifically, name the action "
        "specifically, and state the spatial quadrant precisely "
        "(e.g. 'upper right quadrant', not 'upper area').",
    "next_action":
        "Answer with the single most likely next action only. No explanation.",
    "cvs_assessment":
        "Score the three Critical View of Safety criteria, exact format:\n"
        "Two structures: 0, Cystic plate: 0, Hepatocystic triangle: 0\n"
        "0 = not achieved, 1 = partially achieved, 2 = fully achieved. "
        "Intermediate scores of 1 are common — do not default to all-0 or all-2.",
    "skill_assessment":
        "Rate all six OSATS dimensions from 1 to 5, in this exact format and "
        "order:\n"
        "Respect for tissue: 4/5, Suture/needle handling: 4/5, Time and "
        "motion: 3/5, Flow of operation: 4/5, Overall performance: 4/5, "
        "Quality of final product: 4/5\n"
        "Use the full range; 3 and 4 are the most common ratings.",
}

PRECISION_RULES = (
    "Rules:\n"
    "- Never write 'tool', 'instrument', 'tissue', 'structure' or 'area' when a "
    "specific name is identifiable. Vague wording is scored as an error.\n"
    "- Use the anatomical and instrument vocabulary of this dataset.\n"
    "- Describe only what is visible in the supplied frames. Do not infer steps "
    "you cannot see."
)


def timestamp_grid(times: list[float], max_show: int = 96) -> str:
    if not times:
        return ""
    if len(times) <= max_show:
        ts = ", ".join(f"{t:.1f}" for t in times)
    else:
        step = len(times) / max_show
        ts = ", ".join(f"{times[int(i * step)]:.1f}" for i in range(max_show)) + ", ..."
    return (f"You are given {len(times)} frames sampled non-uniformly from the "
            f"clip. Their timestamps in seconds, in order, are: [{ts}]. "
            f"The clip spans {times[0]:.1f}s to {times[-1]:.1f}s. All times in "
            f"your answer must be in these clip-relative seconds.")


def region_hint(sample) -> str:
    rc = sample.get("RC_info") or {}
    box = rc.get("start_frame_bbox")
    if not box:
        return ""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (f"The target region is the box [{int(x1)}, {int(y1)}, {int(x2)}, "
            f"{int(y2)}] (centre {int(cx)},{int(cy)}) on the frame highlighted "
            f"below. Track that object through the interval.")


def build_system(sample, ctcd_context: str | None = None) -> str:
    task = normalize_task(sample["qa_type"])
    parts = [BASE_ROLE]
    hint = dataset_hint(sample.get("dataset_name", ""))
    if hint:
        parts.append(hint)
    parts.append(PRECISION_RULES)
    parts.append("Output format:\n" + OUTPUT_SPEC.get(task, ""))
    if ctcd_context:
        parts.append(
            "Additional context derived from your own analysis of other "
            "questions about this same video (treat as a strong prior, but "
            "override it if the frames contradict it):\n" + ctcd_context)
    return "\n\n".join(p for p in parts if p)


def action_vocab(sample) -> str:
    """struc_info carries the closed action list for this procedure. Handing it
    to the model turns open-ended naming into constrained selection, which is
    worth real accuracy on TAL and dense captioning."""
    try:
        from .formats import struc_facts
        f = struc_facts(sample)
    except Exception:
        return ""
    bits = []
    if f.get("procedure"):
        bits.append(f"Procedure: {f['procedure']}.")
    if f.get("action_list"):
        bits.append("The only actions that may be named are: "
                    + "; ".join(f["action_list"]) + ".")
    return " ".join(bits)


def build_user(sample, times: list[float]) -> str:
    """Original question + our injected temporal/spatial scaffolding."""
    q = next((m["value"] for m in sample["conversations"] if m["from"] == "human"),
             "")
    q = q.replace("<video>", "").strip()
    blocks = [timestamp_grid(times)]
    av = action_vocab(sample)
    if av:
        blocks.append(av)
    rh = region_hint(sample)
    if rh:
        blocks.append(rh)
    blocks.append(q)
    return "\n\n".join(b for b in blocks if b)


def get_answer(sample) -> str | None:
    return next((m["value"] for m in sample["conversations"] if m["from"] == "gpt"),
                None)
