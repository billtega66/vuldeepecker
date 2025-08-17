# plot_confusion.py
import os, json
import numpy as np
import matplotlib.pyplot as plt

MODEL_DIR = os.path.dirname(__file__)  # adjust if needed
METRICS_PATH = os.path.join(MODEL_DIR, "model", "eval_metrics.json")  # <-- change if needed

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    m = json.load(f)

cm = np.array(m["confusion_matrix"], dtype=int)
labels = ["Safe (0)", "Vulnerable (1)"]

plt.figure(figsize=(4.8, 4.5))
im = plt.imshow(cm, aspect="equal")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks([0,1], ["Pred 0", "Pred 1"])
plt.yticks([0,1], ["True 0", "True 1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.title("Confusion Matrix")
plt.tight_layout()
out_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(out_path, dpi=150)
print("Saved:", out_path)
