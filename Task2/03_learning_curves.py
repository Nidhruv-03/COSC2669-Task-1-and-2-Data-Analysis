"""
Task 2 Part 2 - Q2: model performance under varying training set sizes.

Dataset 1 uses GroupKFold on AccountID so that the account-level separation
established in Task 1 is preserved at every training-set size.
Dataset 2 uses StratifiedKFold to hold the 10 Category proportions stable.
Scoring is macro-F1 (the tuning metric), not accuracy, so the curve is
comparable to the reported model-selection scores.
"""
import joblib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, GroupKFold, StratifiedKFold
from sklearn.base import clone

s = joblib.load("state.joblib")
frac = np.linspace(0.1, 1.0, 10)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
results = {}

for ax, key in zip(axes, ["d1", "d2"]):
    if key == "d1":
        X = s["X1"].iloc[s["tr_idx"]]; y = s["y1"].iloc[s["tr_idx"]]
        grp = s["groups"].iloc[s["tr_idx"]]
        est, cv, kw = clone(s["dt"]), GroupKFold(n_splits=5), dict(groups=grp)
        title, base = "Dataset 1 — Tuned Decision Tree (Channel)", 1/3
    else:
        X = s["X2f"].loc[s["tr2"]]; y = s["y2"].loc[s["tr2"]]
        est, kw = clone(s["rf"]), {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        title, base = "Dataset 2 — Tuned Random Forest (Category)", 1/10

    sizes, tr, va = learning_curve(est, X, y, train_sizes=frac, cv=cv,
                                   scoring="f1_macro", n_jobs=-1,
                                   shuffle=True, random_state=42, **kw)
    tm, ts = tr.mean(1), tr.std(1)
    vm, vs = va.mean(1), va.std(1)
    results[key] = (sizes, tm, ts, vm, vs)

    ax.plot(sizes, tm, "o-", color="#1f77b4", label="Training macro-F1")
    ax.fill_between(sizes, tm - ts, tm + ts, alpha=.15, color="#1f77b4")
    ax.plot(sizes, vm, "s-", color="#d62728", label="Cross-validation macro-F1")
    ax.fill_between(sizes, vm - vs, vm + vs, alpha=.15, color="#d62728")
    ax.axhline(base, ls="--", lw=1, color="grey",
               label=f"Random-guess macro-F1 ({base:.2f})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Training examples"); ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0, 0.75); ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig("figures/learning_curves.png", dpi=200, bbox_inches="tight")

for key, label in [("d1", "DATASET 1 (Decision Tree)"), ("d2", "DATASET 2 (Random Forest)")]:
    sizes, tm, ts, vm, vs = results[key]
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"{'n_train':>8}{'train F1':>11}{'CV F1':>9}{'CV sd':>8}{'gap':>8}")
    for n, a, b, sd in zip(sizes, tm, vm, vs):
        print(f"{n:>8.0f}{a:>11.3f}{b:>9.3f}{sd:>8.3f}{a-b:>8.3f}")
    # slope over the final 30% of the curve
    k = max(2, len(sizes)//3)
    slope = np.polyfit(sizes[-k:], vm[-k:], 1)[0]
    print(f"\nCV-score slope over last {k} points: {slope*1000:+.4f} macro-F1 per 1000 extra rows")
    print(f"final train-CV gap: {tm[-1]-vm[-1]:.3f}")

print("\nsaved figures/learning_curves.png")
