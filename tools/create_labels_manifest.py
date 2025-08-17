# tools/create_labels_manifest.py
import os, re, json, sys

# path shim for utils.config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

ROOT = CONFIG["code_dir"]
OUT  = CONFIG["labels_manifest"]

# -------- patterns we want to detect in source text --------
# Juliet/SARD naming conventions
RE_CWE_BAD_FN   = re.compile(r"\bCWE\d+_[A-Za-z0-9_]*_bad\b", re.I)
RE_CWE_GOOD_FN  = re.compile(r"\bCWE\d+_[A-Za-z0-9_]*_good(?:G2B|B2G)?\b", re.I)
# generic “good/bad” function ids
RE_FN_BAD       = re.compile(r"\b[A-Za-z_]\w*_(?:bad|vuln)\b", re.I)
RE_FN_GOOD      = re.compile(r"\b[A-Za-z_]\w*_(?:good|safe)\b", re.I)
# comments/macros often found in Juliet
RE_BAD_COMMENT  = re.compile(r"/\*\s*BAD\s*\*/", re.I)
RE_GOOD_COMMENT = re.compile(r"/\*\s*GOOD\s*\*/", re.I)
RE_OMITBAD      = re.compile(r"\bOMITBAD\b")
RE_OMITGOOD     = re.compile(r"\bOMITGOOD\b")

# filename/dir fallbacks
RE_DIR_BAD   = re.compile(r"(^|[/_.-])(bad|vul|vuln|vulnerable|unsafe)([/_.-]|$)", re.I)
RE_DIR_GOOD  = re.compile(r"(^|[/_.-])(good|safe|benign|clean)([/_.-]|$)", re.I)

def sniff_file(path: str):
    """Return 1 (vul), 0 (safe), 'both', or None."""
    base   = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    # quick filename/dir hints
    hint_bad  = RE_DIR_BAD.search(base) or RE_DIR_BAD.search(parent)
    hint_good = RE_DIR_GOOD.search(base) or RE_DIR_GOOD.search(parent)
    if hint_bad and not hint_good:  return 1
    if hint_good and not hint_bad:  return 0

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read(800_000)  # cap ~800KB
    except Exception:
        return None

    has_bad  = any(rx.search(text) for rx in (RE_CWE_BAD_FN, RE_FN_BAD, RE_BAD_COMMENT))
    has_good = any(rx.search(text) for rx in (RE_CWE_GOOD_FN, RE_FN_GOOD, RE_GOOD_COMMENT))

    # Sometimes Juliet uses macros; they don’t directly imply labels but often co-occur.
    # We don't force-label from OMIT*, but they can help you inspect later.

    if has_bad and has_good: return "both"
    if has_bad:              return 1
    if has_good:             return 0
    return None

def main():
    labels = {}
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            full = os.path.join(dirpath, f)
            rel  = os.path.relpath(full, ROOT).replace("\\", "/")
            labels[rel] = sniff_file(full)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)

    print(f"Wrote manifest with {len(labels)} entries to {OUT}")
    print("Counts:",
          sum(v == 0 for v in labels.values()), "safe;",
          sum(v == 1 for v in labels.values()), "vul;",
          sum(v == 'both' for v in labels.values()), "both;",
          sum(v is None for v in labels.values()), "unknown")

if __name__ == "__main__":
    main()
