import os, sys, json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import CONFIG

ROOT = CONFIG["code_dir"]
MF   = CONFIG["labels_manifest"]

def apply(folder_rel: str, value):
    with open(MF, "r", encoding="utf-8") as f:
        labels = json.load(f)
    folder_rel = folder_rel.strip("/")

    count = 0
    for rel in list(labels.keys()):
        if rel.startswith(folder_rel + "/") or rel == folder_rel:
            labels[rel] = value
            count += 1

    with open(MF, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Set {count} files under '{folder_rel}' to {value}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m tools.label_subtree <folder_rel_to_code_dir> <0|1|both>")
        sys.exit(1)
    apply(sys.argv[1], int(sys.argv[2]) if sys.argv[2] in ("0","1") else sys.argv[2])