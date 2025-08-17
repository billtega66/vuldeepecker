# plot_per_cwe.py
import os, json
import matplotlib.pyplot as plt
import numpy as np

# Point this at the same folder where eval_metrics.json was written
MODEL_DIR = os.path.dirname(__file__)  # adjust if needed
METRICS_PATH = os.path.join(MODEL_DIR, "model", "eval_metrics.json")  # <-- change if your path differs

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

per_cwe = metrics.get("per_cwe", {})
if not per_cwe:
    raise SystemExit("No per_cwe stats found in eval_metrics.json. Re-run evaluate_model once that code is active.")

cwe_ids = list(per_cwe.keys())
P = [per_cwe[c]["precision"] for c in cwe_ids]
R = [per_cwe[c]["recall"]    for c in cwe_ids]
F = [per_cwe[c]["f1"]        for c in cwe_ids]

x = np.arange(len(cwe_ids))
w = 0.25

plt.figure(figsize=(8, 4.5))
plt.bar(x - w, P, w, label="Precision")
plt.bar(x,     R, w, label="Recall")
plt.bar(x + w, F, w, label="F1")
plt.xticks(x, cwe_ids, rotation=0)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Per-CWE Precision / Recall / F1 (Vulnerable class)")
plt.legend()
out_path = os.path.join(MODEL_DIR, "per_cwe_metrics.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print("Saved:", out_path)
