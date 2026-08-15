"""
Canonical Surgical Ontology (CSO)
=================================
Contribution #3. Eight datasets, eight dialects. CholecT50 says "grasper",
CoPESD says "forceps", EgoSurgery says "Maryland dissector" - same referent.

Used in three places:
  1. prompts.py     -> dataset-conditioned vocabulary hints (say it the way the
                       source dataset says it, because the LLM judge scores
                       *similarity to the reference*, not absolute correctness)
  2. postprocess.py -> terminology repair (map generic -> specific for the dataset)
  3. rewards.py     -> terminology-F1 shaping term for GRPO

Run `python -m src.ontology --audit <train.json>` to find frequent surface forms
in the GT that are NOT yet covered - that tells exactly what to add.
"""
from __future__ import annotations
import re, json, argparse, collections

# ---------------------------------------------------------------- instruments
# canonical_id -> (surface aliases, preferred surface form per dataset)
INSTRUMENTS = {
    "grasper":      ["grasper", "graspers", "grasping forceps", "atraumatic grasper",
                     "fenestrated grasper", "prograsp", "tissue forceps"],
    "maryland":     ["maryland", "maryland dissector", "maryland forceps",
                     "dissecting forceps", "curved dissector"],
    "bipolar":      ["bipolar", "bipolar forceps", "bipolar instrument",
                     "bipolar coagulator", "bipolar electrode"],
    "hook":         ["hook", "l-hook", "electrocautery hook", "cautery hook",
                     "monopolar hook", "diathermy hook"],
    "scissors":     ["scissors", "metzenbaum", "curved scissors", "mayo scissors",
                     "monopolar scissors"],
    "clipper":      ["clipper", "clip applier", "clip applicator", "hem-o-lok"],
    "irrigator":    ["irrigator", "suction", "suction irrigator", "aspirator",
                     "suction device"],
    "specimen_bag": ["specimen bag", "endobag", "retrieval bag", "extraction bag"],
    "needle_driver": ["needle driver", "needle holder", "castroviejo"],
    "snare":        ["snare", "polypectomy snare"],
    "it_knife":     ["it knife", "it-knife", "insulated tip knife", "itknife"],
    "dual_knife":   ["dual knife", "dualknife", "dual-knife"],
    "hook_knife":   ["hook knife", "hookknife"],
    "injection_needle": ["injection needle", "injector", "submucosal injection needle"],
    "endoscope":    ["endoscope", "scope", "gastroscope", "colonoscope", "camera"],
    "trocar":       ["trocar", "port", "cannula"],
    "syringe":      ["syringe"],
    "swab":         ["swab", "alcohol swab", "alcohol pad", "gauze", "cotton ball"],
    "gloves":       ["gloves", "glove"],
    "forceps_open": ["forceps", "hemostat", "kelly clamp", "clamp", "retractor"],
    "suture":       ["suture", "stitch", "thread", "silk"],
    "scalpel":      ["scalpel", "knife", "blade", "bovie", "electrocautery pencil"],
}

# ------------------------------------------------------------------ actions
VERBS = {
    "grasp":     ["grasp", "grasps", "grasping", "grip", "grips", "gripping",
                  "hold", "holds", "holding", "seize"],
    "retract":   ["retract", "retracts", "retracting", "pull", "pulls", "lift",
                  "lifts", "elevate", "elevates", "counter-traction", "traction"],
    "dissect":   ["dissect", "dissects", "dissecting", "dissection", "separate",
                  "separates", "peel", "peels", "mobilize", "mobilizes"],
    "coagulate": ["coagulate", "coagulates", "coagulating", "cauterize",
                  "cauterizes", "electrocautery", "hemostasis", "burn", "seal"],
    "clip":      ["clip", "clips", "clipping", "apply clip", "applies clips"],
    "cut":       ["cut", "cuts", "cutting", "transect", "transects", "incise",
                  "incises", "divide", "divides", "sever"],
    "aspirate":  ["aspirate", "aspirates", "aspirating", "suction", "suctions",
                  "evacuate"],
    "irrigate":  ["irrigate", "irrigates", "irrigating", "wash", "lavage", "flush"],
    "pack":      ["pack", "packs", "packing", "place in bag", "bag"],
    "inject":    ["inject", "injects", "injecting", "injection", "puncture",
                  "punctures", "insert needle"],
    "suture":    ["suture", "sutures", "suturing", "stitch", "stitches", "sew"],
    "tie":       ["tie", "ties", "tying", "knot", "knots", "ligate", "ligates"],
    "disinfect": ["disinfect", "disinfects", "disinfecting", "sterilize", "swab",
                  "clean", "cleans", "antiseptic", "prep"],
    "insert":    ["insert", "inserts", "inserting", "advance", "advances"],
    "withdraw":  ["withdraw", "withdraws", "remove", "removes", "extract",
                  "extracts", "retrieve"],
    "idle":      ["idle", "null", "no action", "stationary", "rest"],
}

# ------------------------------------------------------------------ anatomy
ANATOMY = {
    "gallbladder":       ["gallbladder", "gall bladder", "gb"],
    "cystic_duct":       ["cystic duct"],
    "cystic_artery":     ["cystic artery"],
    "cystic_plate":      ["cystic plate", "gallbladder bed", "liver bed",
                          "gallbladder fossa"],
    "cystic_pedicle":    ["cystic pedicle"],
    "hepatocystic_triangle": ["hepatocystic triangle", "calot's triangle",
                              "calot triangle", "triangle of calot"],
    "common_bile_duct":  ["common bile duct", "cbd", "bile duct"],
    "liver":             ["liver", "hepatic surface", "hepatic parenchyma"],
    "peritoneum":        ["peritoneum", "peritoneal", "serosa"],
    "omentum":           ["omentum", "omental", "fatty tissue"],
    "adhesion":          ["adhesion", "adhesions", "fibrous band"],
    "abdominal_wall":    ["abdominal wall", "abdominal cavity", "cavity"],
    "blood_vessel":      ["blood vessel", "vessel", "artery", "vein"],
    "blood":             ["blood", "bleeding", "hemorrhage", "clot"],
    "fluid":             ["fluid", "bile", "irrigation fluid", "effusion"],
    "gut":               ["gut", "bowel", "intestine", "duodenum", "colon"],
    "mucosa":            ["mucosa", "mucosal layer", "mucosal flap"],
    "submucosa":         ["submucosa", "submucosal layer", "submucosal space"],
    "muscle_layer":      ["muscle layer", "muscularis", "muscularis propria"],
    "lesion":            ["lesion", "polyp", "tumor", "neoplasm", "mass"],
    "skin":              ["skin", "epidermis", "dermis", "cutaneous"],
    "forearm":           ["forearm", "volar aspect", "arm", "antecubital"],
    "wound":             ["wound", "incision", "incision site"],
    "fascia":            ["fascia", "fascial layer"],
    "specimen":          ["specimen", "resected tissue", "sample"],
}

# ---------------------------------------------------- dataset preferred dialect
# When the model must name something, prefer the surface form the source dataset
# uses, because the LLM judge grades similarity to that dataset's reference.
DATASET_DIALECT = {
    "CholecT50":     {"vocab": ["grasper", "bipolar", "hook", "scissors", "clipper",
                                "irrigator"],
                      "note": "Use CholecT50 triplet vocabulary: "
                              "<instrument, verb, target>. Instruments are exactly: "
                              "grasper, bipolar, hook, scissors, clipper, irrigator."},
    "CholecTrack20": {"vocab": ["grasper", "bipolar", "hook", "scissors", "clipper",
                                "irrigator"],
                      "note": "Laparoscopic cholecystectomy, tool-tracking source. "
                              "Name the specific instrument and its spatial quadrant."},
    "Cholec80_CVS":  {"vocab": ["cystic duct", "cystic artery", "cystic plate",
                                "hepatocystic triangle"],
                      "note": "Critical View of Safety assessment. Score each "
                              "criterion 0 (not achieved), 1 (partial), 2 (achieved)."},
    "CoPESD":        {"vocab": ["forceps", "it knife", "dual knife", "snare",
                                "submucosa", "mucosa", "muscle layer"],
                      "note": "Endoscopic submucosal dissection (ESD). Use ESD "
                              "vocabulary: submucosal injection, mucosal incision, "
                              "submucosal dissection, traction."},
    "EgoSurgery":    {"vocab": ["forceps", "scissors", "needle driver", "gauze",
                                "scalpel"],
                      "note": "Egocentric open surgery. Head-mounted view; hands "
                              "and instruments enter from the bottom of frame."},
    "AVOS":          {"vocab": ["cutting", "tying", "suturing"],
                      "note": "Open surgery from video. Actions are limited to: "
                              "cutting, tying, suturing."},
    "JIGSAWS":       {"vocab": ["needle driver", "grasper", "needle", "suture"],
                      "note": "Robotic surgery (da Vinci). Suturing / knot-tying / "
                              "needle-passing tasks; grade with OSATS dimensions."},
    "NurViD":        {"vocab": ["syringe", "swab", "gloves", "gauze"],
                      "note": "Nursing procedure in a clinical/simulation setting. "
                              "Name the nursing action explicitly."},
}

CATEGORIES = {"instrument": INSTRUMENTS, "verb": VERBS, "anatomy": ANATOMY}

# --------------------------------------------------------------- lookup index
_ALIAS2ID: dict[str, tuple[str, str]] = {}
for _cat, _table in CATEGORIES.items():
    for _cid, _aliases in _table.items():
        for _a in _aliases:
            _ALIAS2ID.setdefault(_a.lower(), (_cat, _cid))
# longest alias first so "cystic duct" beats "duct"
_ALIASES_SORTED = sorted(_ALIAS2ID, key=len, reverse=True)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _ALIASES_SORTED) + r")\b", re.I
)

# generic terms we want to *avoid* producing - the judge penalises vagueness (R3)
VAGUE_TERMS = {
    "tool", "instrument", "device", "object", "tissue", "structure", "area",
    "region", "thing", "surface", "something", "part", "material",
}


def extract(text: str) -> dict[str, set[str]]:
    """Return {'instrument': {...}, 'verb': {...}, 'anatomy': {...}} canonical ids."""
    out = {"instrument": set(), "verb": set(), "anatomy": set()}
    if not text:
        return out
    for m in _PATTERN.finditer(text):
        cat, cid = _ALIAS2ID[m.group(1).lower()]
        out[cat].add(cid)
    return out


def terminology_f1(pred: str, ref: str, weights=(1.0, 0.8, 1.0)) -> float:
    """Weighted F1 over canonical (instrument, verb, anatomy) sets.

    This is the reward-shaping proxy for the judge's R1/R2/R5 dimensions.
    Cheap, deterministic, no API call. Correlates well enough to steer RL.
    """
    p, r = extract(pred), extract(ref)
    num = den_p = den_r = 0.0
    for w, cat in zip(weights, ("instrument", "verb", "anatomy")):
        num += w * len(p[cat] & r[cat])
        den_p += w * len(p[cat])
        den_r += w * len(r[cat])
    if den_p == 0 or den_r == 0:
        return 0.0
    prec, rec = num / den_p, num / den_r
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def vagueness_penalty(text: str) -> float:
    """Fraction of vague tokens. Subtract from reward (judge dimension R3)."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    if not toks:
        return 1.0
    return sum(t in VAGUE_TERMS for t in toks) / len(toks)


def preferred_surface(canonical_id: str) -> str:
    for table in CATEGORIES.values():
        if canonical_id in table:
            return table[canonical_id][0]
    return canonical_id


def dataset_hint(dataset_name: str) -> str:
    d = DATASET_DIALECT.get(dataset_name)
    return d["note"] if d else ""


# ---------------------------------------------------------------- audit tool
def audit(train_json: str, topk: int = 60):
    """Find frequent GT nouns/verbs NOT covered by the ontology. Add them."""
    with open(train_json) as f:
        data = json.load(f)
    cnt = collections.Counter()
    stop = set("""the a an of in on to and or is are was were be been with for at
        by from as that this these those it its his her their they we you i not
        do does did has have had will would can could shows video frame frames
        seconds second while during after before then also which where when what
        into over under above below left right upper lower top bottom side""".split())
    for s in data:
        gt = next((m["value"] for m in s["conversations"] if m["from"] == "gpt"), "")
        covered = {m.group(1).lower() for m in _PATTERN.finditer(gt)}
        covered_words = set(" ".join(covered).split())
        for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", gt.lower()):
            if w not in stop and w not in covered_words:
                cnt[w] += 1
    print(f"Top {topk} uncovered terms in ground truth - consider adding:")
    for w, c in cnt.most_common(topk):
        print(f"  {c:6d}  {w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", nargs="?", const="", default=None,
                    help="path to trainval json (defaults to configs/paths.yaml)")
    a = ap.parse_args()
    if a.audit is not None:
        from .paths import need
        audit(a.audit or need("trainval_json"))
    else:
        demo_ref = "The grasper retracts the gallbladder to expose the cystic duct."
        demo_bad = "The tool grabs tissue in the upper area."
        print("ref  ->", extract(demo_ref))
        print("pred ->", extract(demo_bad))
        print("term-F1:", round(terminology_f1(demo_bad, demo_ref), 3))
        print("vagueness:", round(vagueness_penalty(demo_bad), 3))
