"""
Task 2 Part 2 - Q1: how optimistic is the reported cross-validation score?

In Task 1 the same GridSearchCV folds were used BOTH to select hyperparameters
and to report the cross-validated score (grid.best_score_). Selecting the
maximum over 72 (Dataset 1) / 96 (Dataset 2) configurations on the same folds
biases that maximum upwards: it partly measures which configuration best fits
the fold noise, not which generalises.

Nested CV separates the two. An inner loop selects hyperparameters; an
untouched outer fold scores the selected model. The difference between the
non-nested and nested scores is the optimism.
"""
import joblib, numpy as np
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     GroupKFold, StratifiedKFold)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RS = 42
s = joblib.load("state.joblib")

# ------------------------------------------------------------------ Dataset 1
X1 = s["X1"].iloc[s["tr_idx"]]; y1 = s["y1"].iloc[s["tr_idx"]]
g1 = s["groups"].iloc[s["tr_idx"]]
pipe1 = Pipeline([("pre", clone(s["pre1"])),
                  ("clf", DecisionTreeClassifier(random_state=RS))])
grid1 = {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_split": [10, 20, 50],
         "clf__min_samples_leaf": [5, 10, 20], "clf__criterion": ["gini", "entropy"]}

inner1 = GridSearchCV(pipe1, grid1, scoring="f1_macro",
                      cv=GroupKFold(n_splits=4), n_jobs=-1)
non_nested1 = GridSearchCV(pipe1, grid1, scoring="f1_macro",
                           cv=GroupKFold(n_splits=5), n_jobs=-1
                           ).fit(X1, y1, groups=g1).best_score_
nested1 = cross_val_score(inner1, X1, y1, groups=g1, scoring="f1_macro",
                          cv=GroupKFold(n_splits=5), n_jobs=-1,
                          params={"groups": g1})

# ------------------------------------------------------------------ Dataset 2
X2 = s["X2f"].loc[s["tr2"]]; y2 = s["y2"].loc[s["tr2"]]
pipe2 = Pipeline([("pre", clone(s["pre2"])),
                  ("clf", RandomForestClassifier(random_state=RS, n_jobs=1))])
grid2 = {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, 15, None],
         "clf__min_samples_split": [5, 10, 20], "clf__min_samples_leaf": [2, 5, 10],
         "clf__max_features": ["sqrt"]}

inner2 = GridSearchCV(pipe2, grid2, scoring="f1_macro",
                      cv=StratifiedKFold(4, shuffle=True, random_state=RS), n_jobs=-1)
non_nested2 = GridSearchCV(pipe2, grid2, scoring="f1_macro",
                           cv=StratifiedKFold(5, shuffle=True, random_state=RS),
                           n_jobs=-1).fit(X2, y2).best_score_
nested2 = cross_val_score(inner2, X2, y2, scoring="f1_macro",
                          cv=StratifiedKFold(5, shuffle=True, random_state=RS), n_jobs=-1)

for label, nn, nest, grid_size in [
        ("DATASET 1 - Decision Tree (Channel)",   non_nested1, nested1, 72),
        ("DATASET 2 - Random Forest (Category)",  non_nested2, nested2, 96)]:
    print(f"\n{'='*66}\n{label}\n{'='*66}")
    print(f"configurations searched            : {grid_size}")
    print(f"non-nested CV macro-F1 (reported)  : {nn:.3f}")
    print(f"nested CV macro-F1 (unbiased)      : {nest.mean():.3f} "
          f"(sd {nest.std():.3f})")
    print(f"optimism (non-nested - nested)     : {nn - nest.mean():+.3f}")
    print(f"outer fold scores                  : {np.round(nest, 3)}")
    print(f"spread across outer folds          : {nest.max() - nest.min():.3f}")
