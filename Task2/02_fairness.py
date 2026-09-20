"""
Task 2 Part 2 - Q3: Bias analysis with Microsoft Fairlearn (Dataset 1).

Tool choice rationale: Dataset 2 contains no demographic attributes at all
(Date, Description, Category, Amount, Type), so a fairness audit is not
possible there. Dataset 1 contains CustomerAge and CustomerOccupation, and
BOTH are inputs to the tuned Decision Tree - so the model is demonstrably
using protected-type attributes to allocate predictions.

Fairlearn's demographic_parity_difference / equalized_odds_difference are
binary-only. Channel has 3 classes. Two-part approach:
  (a) MetricFrame for per-group accuracy / macro-F1 / selection rate (multiclass-safe)
  (b) one-vs-rest binarisation per channel for the parity metrics
"""
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from fairlearn.metrics import (MetricFrame, selection_rate,
                               demographic_parity_difference,
                               equalized_odds_difference)

s = joblib.load("state.joblib")
d1, X1, y1, te_idx, dt = s["d1"], s["X1"], s["y1"], s["te_idx"], s["dt"]

X1te, y1te = X1.iloc[te_idx], y1.iloc[te_idx]
y_pred = dt.predict(X1te)

age_band = pd.cut(d1["CustomerAge"].iloc[te_idx], [17, 29, 44, 59, 120],
                  labels=["18-29", "30-44", "45-59", "60+"])
occ = d1["CustomerOccupation"].iloc[te_idx]

metrics = {
    "count":     lambda yt, yp: len(yt),
    "accuracy":  accuracy_score,
    "macro_F1":  lambda yt, yp: f1_score(yt, yp, average="macro"),
    "ATM_recall":    lambda yt, yp: recall_score(yt, yp, labels=["ATM"],    average="macro", zero_division=0),
    "Branch_recall": lambda yt, yp: recall_score(yt, yp, labels=["Branch"], average="macro", zero_division=0),
    "Online_recall": lambda yt, yp: recall_score(yt, yp, labels=["Online"], average="macro", zero_division=0),
}

for name, sf in [("AGE BAND", age_band), ("OCCUPATION", occ)]:
    mf = MetricFrame(metrics=metrics, y_true=y1te, y_pred=y_pred, sensitive_features=sf)
    print(f"\n{'='*64}\nPER-GROUP PERFORMANCE BY {name}\n{'='*64}")
    print(mf.by_group.round(3).to_string())
    print("\noverall :", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in mf.overall.items()})
    print("max-min gap:")
    for k, v in mf.difference().items():
        if k != "count":
            print(f"   {k:<16} {v:.3f}")

    # (b) one-vs-rest parity metrics
    print(f"\nBinarised parity metrics (one-vs-rest), by {name.lower()}:")
    print(f"   {'channel':<10}{'DP diff':>10}{'EO diff':>10}"
          f"{'sel.rate range':>20}")
    for ch in ["ATM", "Branch", "Online"]:
        yt_b = (y1te == ch).astype(int)
        yp_b = (pd.Series(y_pred, index=y1te.index) == ch).astype(int)
        dp = demographic_parity_difference(yt_b, yp_b, sensitive_features=sf)
        eo = equalized_odds_difference(yt_b, yp_b, sensitive_features=sf)
        sr = MetricFrame(metrics=selection_rate, y_true=yt_b, y_pred=yp_b,
                         sensitive_features=sf).by_group
        print(f"   {ch:<10}{dp:>10.3f}{eo:>10.3f}"
              f"{f'{sr.min():.3f} - {sr.max():.3f}':>20}")

# ---- Is the disparity real signal, or noise? Permutation control. -----------
print(f"\n{'='*64}\nCONTROL: is the age disparity distinguishable from chance?\n{'='*64}")
rng = np.random.default_rng(42)
observed = MetricFrame(metrics=accuracy_score, y_true=y1te, y_pred=y_pred,
                       sensitive_features=age_band).difference()
null = []
for _ in range(2000):
    shuffled = pd.Series(rng.permutation(age_band.values), index=age_band.index)
    null.append(MetricFrame(metrics=accuracy_score, y_true=y1te, y_pred=y_pred,
                            sensitive_features=shuffled).difference())
null = np.array(null)
print(f"observed accuracy gap across age bands : {observed:.3f}")
print(f"gap expected from random group labels  : {null.mean():.3f} "
      f"(95th pctile {np.percentile(null, 95):.3f})")
print(f"permutation p-value                    : {(null >= observed).mean():.3f}")
