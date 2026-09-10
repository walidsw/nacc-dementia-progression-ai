"""
Part B: genetic-risk sensitivity analysis.
Retrain Config 1 (XGBoost, no SMOTE) with all APOE/genetic-risk columns removed.
Everything else (split, preprocessing, class weights, hyperparameters) is held
identical to the main corrected run so the only difference is the input columns.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class Config:
    INPUT_FILE = "output/features_engineered.csv"
    OUTPUT_DIR = "output/"
    ID_COL = "NACCID"
    VISIT_COL = "NACCVNUM"
    CURRENT_TARGET = "target"
    FUTURE_TARGET = "future_target"
    SEED = 42
    N_CLASS = 4
    CLASS_NAMES = ["Normal", "MCI", "Mild", "Severe"]
    WEIGHT_MULTIPLIERS = {0: 1.0, 1: 1.5, 2: 3.0, 3: 2.5}
    XGB_PARAMS = {
        "n_estimators": 1000, "learning_rate": 0.01, "max_depth": 12, "min_child_weight": 1,
        "subsample": 0.9, "colsample_bytree": 0.9, "gamma": 0.1, "reg_alpha": 0.3,
        "reg_lambda": 1.0, "tree_method": "hist", "objective": "multi:softprob",
        "num_class": N_CLASS, "random_state": SEED, "verbosity": 0,
    }


# ---- B0.2: genetic-risk exclusion list (5 raw + 5 engineered + 1 cross-domain) ----
GENETIC_COLS = [
    "NACCAPOE", "NACCNE4S", "NACCFAM", "NACCMOM", "NACCDAD",          # 5 raw
    "apoe4_any", "apoe4_double", "apoe4_count", "apoe4_age_risk", "apoe4_elderly",  # 5 engineered
    "age_apoe4_interaction",                                            # cross-domain back-door term
]


def prepare_data(df):
    y = df[Config.FUTURE_TARGET].copy()
    groups = df[Config.ID_COL].copy()
    X = df.drop(columns=[Config.ID_COL, Config.VISIT_COL, Config.FUTURE_TARGET], errors="ignore")
    return X, y, groups


def main():
    df = pd.read_csv(Config.INPUT_FILE, low_memory=False)
    genetic_cols = [c for c in GENETIC_COLS if c in df.columns]
    print(f"Excluding {len(genetic_cols)} genetic-risk columns: {genetic_cols}", flush=True)
    assert 9 <= len(genetic_cols) <= 12, f"Expected ~10-11 genetic columns, got {len(genetic_cols)}"

    df_ng = df.drop(columns=genetic_cols)

    X, y, groups = prepare_data(df_ng)
    print(f"X shape (no genetic): {X.shape}", flush=True)

    # B0.3: identical patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.SEED)
    tr, te = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
    y_train, y_test = y.iloc[tr].copy(), y.iloc[te].copy()
    n_test, n_pat = len(X_test), groups.iloc[te].nunique()
    print(f"test rows={n_test} test patients={n_pat}", flush=True)
    assert n_test == 29939, f"test rows {n_test} != 29939"
    assert n_pat == 7527, f"test patients {n_pat} != 7527"

    # B0.5: identical class weights (depend only on label distribution y, unchanged)
    classes = np.unique(y)
    base_weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weights = {int(cls): base_weights[i] * Config.WEIGHT_MULTIPLIERS[cls]
                     for i, cls in enumerate(classes)}
    print("class weights:", {k: round(v, 4) for k, v in class_weights.items()}, flush=True)

    # B0.4: identical preprocessing
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)
    X_train = X_train.fillna(-1).replace([np.inf, -np.inf], [1e10, -1e10])
    X_test = X_test.fillna(-1).replace([np.inf, -np.inf], [1e10, -1e10])

    val_size = int(len(X_train) * 0.2)
    X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]
    X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]

    # B0.6: identical model configuration + sample_weight + early stopping
    params = Config.XGB_PARAMS.copy()
    params["early_stopping_rounds"] = 100
    model = xgb.XGBClassifier(**params)
    sw = compute_sample_weight(class_weight=class_weights, y=y_tr)
    print("training (this is the long step)...", flush=True)
    model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_val, y_val)], verbose=False)
    print("training complete", flush=True)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    pc = f1_score(y_test, y_pred, average=None, zero_division=0)
    report = classification_report(y_test, y_pred, target_names=Config.CLASS_NAMES, zero_division=0)

    print(report, flush=True)
    print(f"ACCURACY={acc:.4f} MACRO_F1={f1m:.4f} WEIGHTED_F1={f1w:.4f}", flush=True)

    with open(f"{Config.OUTPUT_DIR}/sensitivity_no_genetic_classification_report.txt", "w") as f:
        f.write("SENSITIVITY ANALYSIS - Config 1 XGBoost WITHOUT genetic/APOE-risk features\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Excluded columns ({len(genetic_cols)}): {genetic_cols}\n")
        f.write(f"Predictors used: {X.shape[1]}\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Overall F1-Macro: {f1m:.4f}\n")
        f.write(f"Overall F1-Weighted: {f1w:.4f}\n\n")
        f.write(report)

    json.dump(
        {"accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w,
         "per_class_f1": {Config.CLASS_NAMES[i]: float(pc[i]) for i in range(Config.N_CLASS)},
         "n_predictors": int(X.shape[1]), "excluded_columns": genetic_cols},
        open(f"{Config.OUTPUT_DIR}/sensitivity_no_genetic_metrics.json", "w"), indent=2)

    np.savetxt(f"{Config.OUTPUT_DIR}/sensitivity_no_genetic_confusion_matrix.csv",
               confusion_matrix(y_test, y_pred), delimiter=",", fmt="%d")
    print("saved outputs. DONE", flush=True)


if __name__ == "__main__":
    main()
