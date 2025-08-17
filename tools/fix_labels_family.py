# tools/fix_labels_family.py
import os, sys, re, json, collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

IN_PATH = CONFIG["labels_manifest"]
OUT_PATH = (IN_PATH if "--inplace" in sys.argv
            else IN_PATH.replace(".json", "_family.json"))

with open(IN_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

def counts(d):
    return {
        "unknown": sum(1 for v in d.values() if v is None),
        "both":    sum(1 for v in d.values() if v == "both"),
        "safe":    sum(1 for v in d.values() if v == 0),
        "vul":     sum(1 for v in d.values() if v == 1),
    }

by_dir = collections.defaultdict(dict)
for rel, lab in labels.items():
    d = os.path.dirname(rel).replace("\\", "/")
    b = os.path.basename(rel)
    by_dir[d][b] = lab

# group by dir + prefix + number (IGNORE extension)
suffix_re = re.compile(r"_(\d+)([a-z])\.(?:c|cpp|h)$", re.I)
bad_re    = re.compile(r"_bad\.(?:c|cpp|h)$", re.I)
good_re   = re.compile(r"_good(?:G2B|B2G)?\.(?:c|cpp|h)$", re.I)

families = collections.defaultdict(dict)
for d, files in by_dir.items():
    for base, lab in files.items():
        m = suffix_re.search(base)
        if not m: continue
        num, letter = m.group(1), m.group(2).lower()
        prefix = base[:m.start()]
        key = (d, prefix, num)  # ext-agnostic
        families[key][letter] = (base, lab)

updates = {}

# pass 1: propagate "both" to null siblings
for (d, prefix, num), members in families.items():
    if any(l == "both" for (_, l) in members.values()):
        for letter, (fname, lab) in members.items():
            if lab is None:
                rel = f"{d}/{fname}" if d else fname
                updates[rel] = "both"

# pass 2: if dir has *_bad.* AND *_good*.*, set any null lettered driver -> "both"
for d, files in by_dir.items():
    if any(bad_re.search(n) for n in files) and any(good_re.search(n) for n in files):
        for base, lab in files.items():
            if lab is None and suffix_re.search(base):
                rel = f"{d}/{base}" if d else base
                updates[rel] = "both"

labels_out = labels.copy()
for rel, v in updates.items():
    if labels_out.get(rel) is None:
        labels_out[rel] = v

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(labels_out, f, indent=2)

print("[fix-labels] input :", IN_PATH)
print("[fix-labels] output:", OUT_PATH)
print("[fix-labels] changes:", len(updates))
print("[fix-labels] before :", counts(labels))
print("[fix-labels] after  :", counts(labels_out))
print("Tip: re-vectorize after this.")
