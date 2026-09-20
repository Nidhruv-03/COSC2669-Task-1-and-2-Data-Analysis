"""
Task 2 Part 2 - Step 1: Reproduce the Task 1 pipeline exactly.

Purpose: before adding learning curves and fairness analysis, confirm the
pipeline reproduces the numbers reported in task1_Nidhruv_Rajagopalan_s4186362.pdf.
Reported: Dataset 1 tuned DT  acc 0.370, macro F1 0.361 (baseline 0.340 / 0.169)
          Dataset 2 tuned RF  acc 0.150, macro F1 0.164 (baseline 0.110 / 0.020)
"""
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import (GroupShuffleSplit, GroupKFold, GridSearchCV,
                                     train_test_split, StratifiedKFold)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RS = 42

# ---------------------------------------------------------------- Dataset 1
df1 = pd.read_csv("data/bank_transactions_data_2.csv")
d1 = df1.copy()
d1["TransactionDate"] = pd.to_datetime(d1["TransactionDate"])
d1["Hour"] = d1["TransactionDate"].dt.hour
d1["DayOfWeek"] = d1["TransactionDate"].dt.day_name()
d1["Month"] = d1["TransactionDate"].dt.month

feat1 = ["TransactionAmount", "TransactionType", "CustomerAge",
         "CustomerOccupation", "AccountBalance", "DayOfWeek", "Month", "Hour"]
num1 = ["TransactionAmount", "CustomerAge", "AccountBalance", "Month", "Hour"]
cat1 = ["TransactionType", "CustomerOccupation", "DayOfWeek"]

X1, y1 = d1[feat1], d1["Channel"]
groups = d1["AccountID"]

pre1 = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), cat1),
                          ("numerical", "passthrough", num1)])

tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                        random_state=RS).split(X1, y1, groups=groups))
X1tr, X1te = X1.iloc[tr_idx], X1.iloc[te_idx]
y1tr, y1te = y1.iloc[tr_idx], y1.iloc[te_idx]
g1tr = groups.iloc[tr_idx]

assert len(set(g1tr) & set(groups.iloc[te_idx])) == 0, "account leakage!"

base1 = Pipeline([("pre", clone(pre1)), ("clf", DummyClassifier(strategy="most_frequent"))])
base1.fit(X1tr, y1tr); bp1 = base1.predict(X1te)

grid1 = GridSearchCV(
    Pipeline([("pre", clone(pre1)), ("clf", DecisionTreeClassifier(random_state=RS))]),
    {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_split": [10, 20, 50],
     "clf__min_samples_leaf": [5, 10, 20], "clf__criterion": ["gini", "entropy"]},
    scoring="f1_macro", cv=GroupKFold(n_splits=5), n_jobs=-1)
grid1.fit(X1tr, y1tr, groups=g1tr)
dt = grid1.best_estimator_
p1 = dt.predict(X1te)

print("=== DATASET 1 (Channel, tuned Decision Tree) ===")
print("best params      :", grid1.best_params_)
print("best CV macro F1 :", round(grid1.best_score_, 3), "  (report: 0.367)")
print("baseline acc/F1  :", round(accuracy_score(y1te, bp1), 3),
      round(f1_score(y1te, bp1, average='macro'), 3), " (report: 0.340 / 0.169)")
print("tuned    acc/F1  :", round(accuracy_score(y1te, p1), 3),
      round(f1_score(y1te, p1, average='macro'), 3), " (report: 0.370 / 0.361)")

# ---------------------------------------------------------------- Dataset 2
df2 = pd.read_csv("data/Personal_Finance_Dataset.csv")
d2 = df2.copy()
d2["Date"] = pd.to_datetime(d2["Date"])
d2["Year"] = d2["Date"].dt.year
d2["Month"] = d2["Date"].dt.month
d2["DayOfWeek"] = d2["Date"].dt.day_name()
d2["DayOfMonth"] = d2["Date"].dt.day
d2["IsWeekend"] = d2["Date"].dt.dayofweek >= 5
d2["LogAmount"] = np.log1p(d2["Amount"])

feat2_init = ["Amount", "LogAmount", "Year", "Month", "DayOfWeek", "DayOfMonth", "IsWeekend"]
X2, y2 = d2[feat2_init], d2["Category"]
X2tr, X2te, y2tr, y2te = train_test_split(X2, y2, test_size=0.20,
                                          random_state=RS, stratify=y2)

feat2 = ["Amount", "Year", "Month", "DayOfWeek", "DayOfMonth", "IsWeekend"]
num2 = ["Amount", "Year", "Month", "DayOfMonth"]
cat2 = ["DayOfWeek", "IsWeekend"]
X2f = d2[feat2]
X2trf, X2tef = X2f.loc[X2tr.index], X2f.loc[X2te.index]

pre2 = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), cat2),
                          ("numerical", "passthrough", num2)])

base2 = Pipeline([("pre", clone(pre2)), ("clf", DummyClassifier(strategy="most_frequent"))])
base2.fit(X2trf, y2tr); bp2 = base2.predict(X2tef)

grid2 = GridSearchCV(
    Pipeline([("pre", clone(pre2)), ("clf", RandomForestClassifier(random_state=RS, n_jobs=-1))]),
    {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10, 15, None],
     "clf__min_samples_split": [5, 10, 20], "clf__min_samples_leaf": [2, 5, 10],
     "clf__max_features": ["sqrt"]},
    scoring="f1_macro", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RS), n_jobs=-1)
grid2.fit(X2trf, y2tr)
rf = grid2.best_estimator_
p2 = rf.predict(X2tef)

print("\n=== DATASET 2 (Category, tuned Random Forest) ===")
print("best params      :", grid2.best_params_)
print("best CV macro F1 :", round(grid2.best_score_, 3), "  (report: 0.182)")
print("baseline acc/F1  :", round(accuracy_score(y2te, bp2), 3),
      round(f1_score(y2te, bp2, average='macro'), 3), " (report: 0.110 / 0.020)")
print("tuned    acc/F1  :", round(accuracy_score(y2te, p2), 3),
      round(f1_score(y2te, p2, average='macro'), 3), " (report: 0.150 / 0.164)")

joblib.dump(dict(d1=d1, X1=X1, y1=y1, groups=groups, tr_idx=tr_idx, te_idx=te_idx,
                 pre1=pre1, dt=dt, best1=grid1.best_params_,
                 d2=d2, X2f=X2f, y2=y2, tr2=X2tr.index, te2=X2te.index,
                 pre2=pre2, rf=rf, best2=grid2.best_params_), "state.joblib")
print("\nstate saved.")
