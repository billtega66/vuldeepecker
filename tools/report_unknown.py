# tools/report_unknown.py
import os, sys, json, re, collections
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

IN_PATH = CONFIG["labels_manifest"]
ROOT = CONFIG["code_dir"]

RE_ANY_GOODBAD = re.compile(r'(bad|good|goodG2B|goodB2G)', re.I)

with open(IN_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

unknowns = [rel for rel,v in labels.items() if v is None]
print(f"[report] unknown count: {len(unknowns)}")

by_dir = collections.defaultdict(list)
for rel in unknowns:
    by_dir[os.path.dirname(rel)].append(rel)

# top 20 dirs by unknown count
top = sorted(by_dir.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]
for d, files in top:
    print(f"\n[{len(files)}] {d or '.'}")
    # sample reasons for up to 8 files
    for rel in files[:8]:
        fp = os.path.join(ROOT, rel)
        ext = os.path.splitext(rel)[1].lower()
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(40000)
        except Exception:
            head = ""
        reason = []
        if ext not in (".c", ".cpp", ".h"):
            reason.append("non-C/C++ file")
        if not RE_ANY_GOODBAD.search(head):
            reason.append("no good/bad markers")
        if "#ifndef" in head and "#define" in head and ext == ".h":
            reason.append("header-only (likely interface)")
        print(f"  - {rel} :: " + (", ".join(reason) or "needs manual check"))
