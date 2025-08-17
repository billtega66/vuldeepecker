import os, json
import matplotlib.pyplot as plt

metrics_dir = "metrics_history"
cwe_scores = {}  # {CWE: {"epoch":[], "precision":[], "recall":[], "f1":[]}}

for file in sorted(os.listdir(metrics_dir)):
    if not file.startswith("eval_metrics_epoch_"):
        continue
    epoch = int(file.split("_")[-1].split(".")[0])
    with open(os.path.join(metrics_dir, file)) as f:
        metrics = json.load(f)
    for cwe, scores in metrics["per_cwe"].items():
        if cwe not in cwe_scores:
            cwe_scores[cwe] = {"epoch": [], "precision": [], "recall": [], "f1": []}
        cwe_scores[cwe]["epoch"].append(epoch)
        cwe_scores[cwe]["precision"].append(scores["precision"])
        cwe_scores[cwe]["recall"].append(scores["recall"])
        cwe_scores[cwe]["f1"].append(scores["f1"])

plt.figure(figsize=(10, 6))
for cwe, vals in cwe_scores.items():
    plt.scatter(vals["epoch"], vals["f1"], label=f"{cwe} F1")
    plt.plot(vals["epoch"], vals["f1"], linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("Per-CWE F1-score Progression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("per_cwe_f1_epochs.png", dpi=150)
print("Saved per_cwe_f1_epochs.png")
