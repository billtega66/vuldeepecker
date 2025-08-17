# tools/fix_labels_code_scan.py
import os, re, json, sys, collections

# allow: python -m tools.fix_labels_code_scan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

ROOT = CONFIG["code_dir"]
IN_PATH = CONFIG["labels_manifest"]
OUT_PATH = (IN_PATH if "--inplace" in sys.argv
            else IN_PATH.replace(".json", "_codescan.json"))

# ---------- Function names (calls & definitions) ----------
# C underscored style
RE_CALL_BAD_US     = re.compile(r'\b[A-Za-z_]\w*_bad\s*\(', re.I)
RE_CALL_GOOD_US    = re.compile(r'\b[A-Za-z_]\w*_good(?:G2B|B2G)?\s*\(', re.I)
RE_DEF_BAD_US      = re.compile(r'\b(?:static\s+)?(?:void|int|long|char|wchar_t)\s+[A-Za-z_]\w*_bad\s*\(', re.I)
RE_DEF_GOOD_US     = re.compile(r'\b(?:static\s+)?(?:void|int|long|char|wchar_t)\s+[A-Za-z_]\w*_good(?:G2B|B2G)?\s*\(', re.I)

# C++ plain names in 62/31/33/… series
RE_CALL_BAD_CXX    = re.compile(r'\bbad\s*\(', re.I)
RE_CALL_GOOD_ANY   = re.compile(r'\bgood(?:G2B|B2G)?\s*\(', re.I)
RE_DEF_BAD_CXX     = re.compile(r'\b(?:void|int|long|char|wchar_t)\s+bad\s*\(', re.I)
RE_DEF_GOOD_ANY    = re.compile(r'\b(?:void|int|long|char|wchar_t)\s+good(?:G2B|B2G)?\s*\(', re.I)

# Juliet sink/source split in b/c/d/e
RE_BAD_SINKSRC     = re.compile(r'\b[A-Za-z_]\w*(?:bad|Bad)(?:Sink|Source)\s*\(', re.I)
RE_GOOD_SINKSRC    = re.compile(r'\b[A-Za-z_]\w*good(?:G2B|B2G)(?:Sink|Source)\s*\(', re.I)

# ---------- Comments ----------
RE_BAD_CMT1        = re.compile(r'/\*\s*FLAW\s*\*/', re.I)
RE_GOOD_CMT1       = re.compile(r'/\*\s*FIX\s*:\s*', re.I)
RE_BAD_CMT2        = re.compile(r'/\*\s*ERROR\s*:', re.I)             # /*ERROR: */
RE_GOOD_CMT2       = re.compile(r'/\*\s*(?:NO\s+ERROR|Tool should not detect)', re.I)
RE_BAD_CMT3        = re.compile(r'/\*\s*POTENTIAL\s+FLAW', re.I)
RE_GOOD_CMT3       = re.compile(r'/\*\s*FIX\s*', re.I)

# ---------- Drivers / macros ----------
RE_OMITBAD         = re.compile(r'\bOMITBAD\b')
RE_OMITGOOD        = re.compile(r'\bOMITGOOD\b')
RE_MAIN            = re.compile(r'\bint\s+main\s*\(', re.I)

# ---------- Dir-level hints ----------
RE_NAME_BAD        = re.compile(r'_bad\.(?:c|cpp|h)$', re.I)
RE_NAME_GOOD       = re.compile(r'_good(?:G2B|B2G)?\.(?:c|cpp|h)$', re.I)

def _read_text(fp: str) -> str | None:
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(800_000)
    except Exception:
        return None

def _dir_has_good_bad(names):
    return any(RE_NAME_BAD.search(n) for n in names) and any(RE_NAME_GOOD.search(n) for n in names)

def _decide_label(text: str, dirnames: list[str]) -> str | int | None:
    if not text:
        return None

    has_bad = any(p.search(text) for p in (
        RE_CALL_BAD_US, RE_DEF_BAD_US, RE_CALL_BAD_CXX, RE_DEF_BAD_CXX,
        RE_BAD_SINKSRC, RE_BAD_CMT1, RE_BAD_CMT2, RE_BAD_CMT3
    ))
    has_good = any(p.search(text) for p in (
        RE_CALL_GOOD_US, RE_DEF_GOOD_US, RE_CALL_GOOD_ANY, RE_DEF_GOOD_ANY,
        RE_GOOD_SINKSRC, RE_GOOD_CMT1, RE_GOOD_CMT2, RE_GOOD_CMT3
    ))

    # driver macro hints
    if RE_MAIN.search(text):
        if RE_CALL_BAD_CXX.search(text) or RE_OMITBAD.search(text):
            has_bad = True
        if RE_CALL_GOOD_ANY.search(text) or RE_OMITGOOD.search(text):
            has_good = True

    if has_bad and has_good:
        return "both"
    if has_bad:
        return 1
    if has_good:
        return 0

    # directory-level: both bad/good implementations exist => mark lettered drivers as both
    if _dir_has_good_bad(dirnames):
        return "both"

    return None

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    by_dir = collections.defaultdict(list)
    for rel in labels:
        d = os.path.dirname(rel).replace("\\", "/")
        by_dir[d].append(os.path.basename(rel))

    before = {
        "unknown": sum(1 for v in labels.values() if v is None),
        "both":    sum(1 for v in labels.values() if v == "both"),
        "safe":    sum(1 for v in labels.values() if v == 0),
        "vul":     sum(1 for v in labels.values() if v == 1),
    }

    updates = {}
    unknowns = [rel for rel,v in labels.items() if v is None]
    for rel in unknowns:
        full = os.path.join(ROOT, rel)
        text = _read_text(full)
        d = os.path.dirname(rel).replace("\\", "/")
        decide = _decide_label(text, by_dir.get(d, []))
        if decide is not None:
            updates[rel] = decide

    labels_out = labels.copy()
    for rel, v in updates.items():
        labels_out[rel] = v

    after = {
        "unknown": sum(1 for v in labels_out.values() if v is None),
        "both":    sum(1 for v in labels_out.values() if v == "both"),
        "safe":    sum(1 for v in labels_out.values() if v == 0),
        "vul":     sum(1 for v in labels_out.values() if v == 1),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_out, f, indent=2)

    print("[codescan] input :", IN_PATH)
    print("[codescan] output:", OUT_PATH)
    print("[codescan] changes:", len(updates))
    print("[codescan] before :", before)
    print("[codescan] after  :", after)
    print("Tip: run fix_labels_family next to propagate to b/c/d/e siblings across .c/.cpp.")

if __name__ == "__main__":
    main()
