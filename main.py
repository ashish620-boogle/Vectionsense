import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_fscore_support
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBClassifier
from joblib import dump, load  # <--- add this
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

seed = 42
MODELS_ML_DIR = "models_ml"
os.makedirs(MODELS_ML_DIR, exist_ok=True)


# =============================================================================
# FILE PATHS (ADAPT TO YOUR LOCAL FOLDER)
# =============================================================================

imu_files = {
    "arpit":      "RC Cybersickness detection - rc_arpit_combined_imu.csv",
    "himanshu":   "RC Cybersickness detection - rc_himanshu_combined_imu.csv",
    "jatin":      "RC Cybersickness detection - rc_jatin_combined_imu.csv",
    "mihir":      "RC Cybersickness detection - rc_mihir_combined_imu.csv",
    "riya":       "RC Cybersickness detection - rc_riya_combined_imu.csv",
    "shreyadeb":  "RC Cybersickness detection - rc_shreyadeb_combined_imu.csv",
    "shubham":    "RC Cybersickness detection - rc_subham_combined_imu.csv",  # note: subham in filename
    "vaishnavi":  "RC Cybersickness detection - rc_vaishnavi_combined_imu.csv",
    "rounak":     "RC Cybersickness detection - rc_rounak_combined_imu.csv",
    "shreya":     "RC Cybersickness detection - rc_shreya_combined_imu.csv",
}

hr_files = {
    "arpit":      "RC Heartrate - arpit_HeartRate.csv",
    "himanshu":   "RC Heartrate - himanshu_HeartRate.csv",
    "jatin":      "RC Heartrate - Jatin_HeartRate.csv",
    "mihir":      "RC Heartrate - Mihir_HeartRate.csv",
    "riya":       "RC Heartrate - Riya_HeartRate.csv",
    "rounak":     "RC Heartrate - Rounak_HeartRate.csv",      # extra, not used in 5/3 split
    "shreya":     "RC Heartrate - Shreya_HeartRate.csv",      # extra, not used in 5/3 split
    "shreyadeb":  "RC Heartrate - Shreyadeb_HeartRate.csv",
    "shubham":    "RC Heartrate - Shubham_HeartRate.csv",
    "vaishnavi":  "RC Heartrate - Vaishnavi_HeartRate.csv",
}

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_series(y, new_len, method="cubic"):
    """
    Interpolate a 1D array y to new_len points using the given method:
      - 'cubic'  : cubic spline (smooth)
      - 'linear' : piecewise linear
      - 'nearest': nearest neighbor
    """
    N = len(y)
    if N < 2 or new_len <= 0:
        return np.array(y, copy=True)

    x_old = np.arange(N)
    x_new = np.linspace(0, N - 1, new_len)

    if method == "cubic":
        cs = CubicSpline(x_old, y)
        return cs(x_new)
    elif method in ("linear", "nearest"):
        f = interp1d(x_old, y, kind=method, fill_value="extrapolate")
        return f(x_new)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

def create_interpolated_hr_csv_for_pid(pid, imu_path, hr_path, out_path, method="cubic"):
    """
    Create a new HRV CSV for one participant where:
      - number of rows == number of IMU rows
      - HR/HRV features are interpolated with the given method:
          'cubic', 'linear', or 'nearest'
      - Cybersickness is aligned by nearest index (no interpolation on labels)
    """
    if not os.path.exists(imu_path):
        print(f"[WARN] IMU file missing for {pid}: {imu_path}")
        return
    if not os.path.exists(hr_path):
        print(f"[WARN] HR file missing for {pid}: {hr_path}")
        return

    # Load data
    imu_df = pd.read_csv(imu_path)
    hr_df = pd.read_csv(hr_path)

    # Ensure label and HR features
    hr_df = ensure_cybersickness_column(hr_df)
    hr_df = drop_non_feature_columns(hr_df, label_col="Cybersickness")
    hr_df = extract_hr_features(hr_df)
    hr_df = drop_non_feature_columns(hr_df, label_col="Cybersickness")

    N_imu = len(imu_df)
    N_hr  = len(hr_df)

    if N_hr < 2:
        print(f"[WARN] Not enough HR samples for interpolation for {pid} (N_hr={N_hr})")
        return

    # Old and new "time" axes (index-based)
    x_old = np.arange(N_hr)
    x_new = np.linspace(0, N_hr - 1, N_imu)

    # Numeric HR/HRV columns (except label)
    numeric_cols = hr_df.select_dtypes(include=[np.number]).columns.tolist()
    if "Cybersickness" in numeric_cols:
        numeric_cols.remove("Cybersickness")

    interp_data = {}

    # Interpolate each numeric HR/HRV feature with the chosen method
    for col in numeric_cols:
        interp_data[col] = interpolate_series(hr_df[col].values, N_imu, method=method)

    # Align Cybersickness by nearest index (no interpolation on labels)
    if "Cybersickness" in hr_df.columns:
        idx_near = np.clip(np.round(x_new).astype(int), 0, N_hr - 1)
        interp_data["Cybersickness"] = hr_df["Cybersickness"].values[idx_near]

    # Optional: carry over IMU time/index
    if "time" in imu_df.columns:
        interp_data["time_imu"] = imu_df["time"].values
    else:
        interp_data["imu_index"] = np.arange(N_imu)

    out_df = pd.DataFrame(interp_data)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {method} interpolated HRV for {pid} to {out_path}")

def create_all_interpolated_hr_csvs():
    """
    For each participant that has both IMU and HR files,
    create interpolated HRV CSVs aligned to IMU length using:
      - cubic spline
      - linear interpolation
      - nearest neighbor
    Output structure:
      hr_aligned/
        cubic/pid_hr_aligned_cubic.csv
        linear/pid_hr_aligned_linear.csv
        nearest/pid_hr_aligned_nearest.csv
    """
    base_dir = "hr_aligned"
    methods = ["cubic", "linear", "nearest"]

    for method in methods:
        out_dir = os.path.join(base_dir, method)
        os.makedirs(out_dir, exist_ok=True)

        for pid in imu_files.keys():
            if pid not in hr_files:
                continue

            imu_path = imu_files[pid]
            hr_path  = hr_files[pid]
            out_path = os.path.join(out_dir, f"{pid}_hr_aligned_{method}.csv")

            create_interpolated_hr_csv_for_pid(pid, imu_path, hr_path, out_path, method=method)


def get_hr_aligned_files(method="cubic"):
    """
    Return hr_files-like dict pointing to hr_aligned/<method>/... CSVs.
    method: 'cubic', 'linear', or 'nearest'
    """
    aligned_root = os.path.join("hr_aligned", method)
    hr_aligned_files = {}
    for pid in imu_files.keys():
        aligned_path = os.path.join(aligned_root, f"{pid}_hr_aligned_{method}.csv")
        if os.path.exists(aligned_path):
            hr_aligned_files[pid] = aligned_path
    return hr_aligned_files



# =============================================================================
# GENERIC HELPERS
# =============================================================================

def train_best_for_fold(X_train, y_train, X_test, y_test):
    """
    Train all classifiers on a fold and return the best one (by accuracy)
    plus its predictions and metrics.
    No plotting here – purely for CV.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = get_classifiers()
    best_name = None
    best_clf = None
    best_metrics = None
    best_y_pred = None
    best_acc = -1.0

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_clf = clf
            best_y_pred = y_pred
            best_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    return best_name, best_clf, scaler, best_y_pred, best_metrics


def ensure_cybersickness_column(df):
    """Ensure label column is exactly named 'Cybersickness'."""
    if "Cybersickness" in df.columns:
        return df
    # auto-detect by 'cyber' substring
    cand = [c for c in df.columns if "cyber" in c.lower()]
    if not cand:
        raise ValueError(f"No Cybersickness-like column found in columns: {df.columns.tolist()}")
    df = df.rename(columns={cand[0]: "Cybersickness"})
    return df

def drop_non_feature_columns(df, label_col="Cybersickness"):
    """
    Drop obvious non-feature columns: comments, filenames, etc.
    Keep label_col.
    """
    cols_to_drop = []
    for c in df.columns:
        cl = c.lower()
        if c == label_col:
            continue
        # obvious meta columns
        if "comment" in cl or "file" in cl or "participant" in cl or "name" in cl:
            cols_to_drop.append(c)

        # time-like columns you don't want as features
        if cl == "time" or cl == "seconds_elapsed":
            cols_to_drop.append(c)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def prepare_modality_arrays(modality_data, pids):
    """
    From {pid: (X, y, ...)} and an ordered pid list,
    build:
      X_all: concatenated features
      y_all: concatenated labels
      groups: array indicating which participant each row came from
    """
    X_list, y_list, g_list = [], [], []

    for i, pid in enumerate(pids):
        X_pid = modality_data[pid][0]
        y_pid = modality_data[pid][1]
        X_list.append(X_pid)
        y_list.append(y_pid)
        g_list.append(np.full(len(y_pid), i))  # group id = participant index

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)
    return X_all, y_all, groups

def cross_validate_modality(modality_data, feature_names, pids, prefix, k_folds=5):
    """
    Perform participant-wise GroupKFold cross-validation for one modality (IMU or HR).
    Uses the same classifiers as train_and_evaluate() and prints mean±std metrics.
    """
    # Build sample-level arrays + group labels (participants)
    X_all, y_all, groups = prepare_modality_arrays(modality_data, pids)

    n_splits = min(k_folds, len(pids))
    gkf = GroupKFold(n_splits=n_splits)

    # metrics[clf_name]["accuracy"] -> [fold1, fold2, ...]
    classifiers_names = list(get_classifiers().keys())
    metrics = {
        name: {"accuracy": [], "precision": [], "recall": [], "f1": []}
        for name in classifiers_names
    }

    fold_idx = 0
    for train_idx, test_idx in gkf.split(X_all, y_all, groups=groups):
        fold_idx += 1
        print(f"\n[{prefix}] Fold {fold_idx}/{n_splits} – train={len(train_idx)}, test={len(test_idx)}")

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # fresh classifiers each fold
        classifiers = get_classifiers()

        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            metrics[name]["accuracy"].append(acc)
            metrics[name]["precision"].append(prec)
            metrics[name]["recall"].append(rec)
            metrics[name]["f1"].append(f1)

    print(f"\n========== {prefix.upper()} – {n_splits}-FOLD GROUPK-FOLD RESULTS ==========")
    for name in classifiers_names:
        m = metrics[name]
        acc_mean, acc_std = np.mean(m["accuracy"]), np.std(m["accuracy"])
        prec_mean, prec_std = np.mean(m["precision"]), np.std(m["precision"])
        rec_mean, rec_std = np.mean(m["recall"]), np.std(m["recall"])
        f1_mean, f1_std = np.mean(m["f1"]), np.std(m["f1"])

        print(f"\n{name}:")
        print(f"  Accuracy : {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"  Precision: {prec_mean:.3f} ± {prec_std:.3f}")
        print(f"  Recall   : {rec_mean:.3f} ± {rec_std:.3f}")
        print(f"  F1-score : {f1_mean:.3f} ± {f1_std:.3f}")

def segment_into_windows(df, label_col="Cybersickness", window_sec=10, sampling_hz=1):
    """
    Segment time-series dataframe into non-overlapping windows.
    Assumes 1 sample per second (sampling_hz=1) unless changed.
    """
    window_size = window_sec * sampling_hz
    rows = len(df)

    X, y = [], []

    for start in range(0, rows, window_size):
        end = start + window_size
        if end > rows:
            break

        chunk = df.iloc[start:end]

        # Use only numeric feature columns (exclude label)
        numeric_cols = chunk.drop(columns=[label_col]).select_dtypes(include=[np.number])
        if numeric_cols.shape[1] == 0:
            continue  # nothing to use

        feature_vector = numeric_cols.mean(axis=0).values

        # Majority vote for label (0/1)
        label = int(chunk[label_col].mean() >= 0.5)

        X.append(feature_vector)
        y.append(label)

    return np.array(X), np.array(y)

def segment_to_sequences(df, label_col="Cybersickness", window_sec=10, sampling_hz=1):
    """
    Return:
      X_seq: list of np.array, each of shape [T, D]  (sequence)
      y_seq: list of int labels (0/1)
    """
    window_size = window_sec * sampling_hz
    rows = len(df)

    X_seq, y_seq = [], []

    for start in range(0, rows, window_size):
        end = start + window_size
        if end > rows:
            break

        chunk = df.iloc[start:end]
        numeric_cols = chunk.drop(columns=[label_col]).select_dtypes(include=[np.number])
        if numeric_cols.shape[1] == 0:
            continue

        # sequence shape: [T, D]
        seq = numeric_cols.values.astype(np.float32)

        # majority label
        label = int(chunk[label_col].mean() >= 0.5)

        X_seq.append(seq)
        y_seq.append(label)

    return X_seq, np.array(y_seq, dtype=np.int64)

def save_all_combined_classifiers(classifiers, scaler, feature_names, train_pids, test_pids, results):
    """
    Save *all* classical ML classifiers that were trained on the Combined (IMU+HR) features.

    classifiers: dict name -> fitted model (from train_and_evaluate)
    scaler:     StandardScaler fitted on combined features
    feature_names: list of combined feature names
    train_pids, test_pids: participants used in this split
    results:    metrics dict per classifier (same keys as from train_and_evaluate)
    """
    for name, clf in classifiers.items():
        metrics = results.get(name, {})
        payload = {
            "mode": "combined",
            "clf_name": name,
            "model": clf,
            "scaler": scaler,
            "feature_names": feature_names,
            "train_pids": train_pids,
            "test_pids": test_pids,
            "metrics": metrics,
        }
        safe_name = name.replace(" ", "_")
        file_name = f"combined_{safe_name}.pkl"
        path = os.path.join(MODELS_ML_DIR, file_name)
        dump(payload, path)
        print(f"[SAVE] Saved combined classifier '{name}' to {path}")


def save_best_ml_model(mode, clf_name, model, scaler, metrics):
    """
    Save the best classical ML model for a given mode (imu/hr/combined)
    so that the transformer script can load and compare later.

    mode: 'imu', 'hr', or 'combined'
    clf_name: classifier name string (e.g. 'Random Forest')
    model: trained sklearn classifier
    scaler: fitted StandardScaler
    metrics: dict with keys ['accuracy', 'precision', 'recall', 'f1']
    """
    payload = {
        "mode": mode,
        "clf_name": clf_name,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }
    safe_name = clf_name.replace(" ", "_")
    file_name = f"best_{mode}_{safe_name}.pkl"
    path = os.path.join(MODELS_ML_DIR, file_name)
    dump(payload, path)
    print(f"[SAVE] Saved best {mode} model ({clf_name}) to {path}")
    return path


# =============================================================================
# IMU FEATURE EXTRACTION
# =============================================================================

def extract_imu_features(df):
    """
    From raw IMU (head + controllers), derive:
    - vel_*, acc_*, jerk_*
    - sway_var
    - trajectory_smoothness
    - movement_energy

    Assumes columns with 'x','y','z' in name are spatial components.
    """
    df = df.copy()

    # Identify base IMU columns (position/orientation components)
    imu_cols = [c for c in df.columns if any(axis in c.lower() for axis in ["x", "y", "z"])]

    # Velocity
    for c in imu_cols:
        df[f"vel_{c}"] = np.gradient(df[c].values)

    # Acceleration
    for c in imu_cols:
        df[f"acc_{c}"] = np.gradient(df[f"vel_{c}"].values)

    # Jerk
    for c in imu_cols:
        df[f"jerk_{c}"] = np.gradient(df[f"acc_{c}"].values)

    # Sway variance across spatial axes
    df["sway_var"] = df[imu_cols].select_dtypes(include=[np.number]).var(axis=1)

    # Trajectory smoothness = norm of jerk vector
    jerk_cols = [c for c in df.columns if c.startswith("jerk_")]
    if jerk_cols:
        df["trajectory_smoothness"] = np.linalg.norm(df[jerk_cols].values, axis=1)
    else:
        df["trajectory_smoothness"] = 0.0

    # Movement energy = sum of squared acceleration components
    acc_cols = [c for c in df.columns if c.startswith("acc_")]
    if acc_cols:
        df["movement_energy"] = (df[acc_cols].values ** 2).sum(axis=1)
    else:
        df["movement_energy"] = 0.0

    # Remove accidental label-derived features if they exist
    bad_cs_cols = ["vel_Cybersickness", "acc_Cybersickness", "jerk_Cybersickness"]
    # print(df.columns)
    present_bad = [c for c in bad_cs_cols if c in df.columns]
    if present_bad:
        df = df.drop(columns=present_bad)

    return df


# =============================================================================
# HR / HRV FEATURE EXTRACTION
# =============================================================================

def extract_hr_features(df):
    """
    Build basic HRV-like features from heart-rate data.

    Assumes:
      - Heart rate column is named 'bpm'
      - Values are in beats per minute

    Features at each sample:
      - hr        : raw bpm (cleaned)
      - hr_diff   : difference between consecutive hr values
      - hr_sq     : squared hr (energy-like)
      - rr        : approx RR interval in seconds (60 / hr)
      - rr_diff   : difference between consecutive rr values
    """
    df = df.copy()

    # Ensure bpm column exists
    if "bpm" not in df.columns:
        raise ValueError(f"'bpm' column not found in columns: {df.columns.tolist()}")

    # Clean heart rate
    df["hr"] = pd.to_numeric(df["bpm"], errors="coerce")\
                   .fillna(method="ffill")\
                   .fillna(method="bfill")

    # Simple HR dynamic features
    df["hr_diff"] = df["hr"].diff().fillna(0.0)
    df["hr_sq"] = df["hr"] ** 2

    # Approximate RR interval from HR (in seconds)
    # RR (sec) = 60 / HR(bpm)
    hr_safe = df["hr"].replace(0, np.nan)
    df["rr"] = (60.0 / hr_safe).fillna(method="ffill").fillna(method="bfill")
    df["rr_diff"] = df["rr"].diff().fillna(0.0)

    return df


# =============================================================================
# TRAINING + EVALUATION (generic, used for IMU and HR)
# =============================================================================

def collate_fn(batch):
    # assuming all sequences have same length T; if not, pad here
    X, y = zip(*batch)
    X = torch.tensor(np.stack(X, axis=0), dtype=torch.float32)  # [B, T, D]
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def get_classifiers():
    return {
        "Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Process": GaussianProcessClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "Neural Net": MLPClassifier(max_iter=500, random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Naive Bayes": GaussianNB(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "XGBoost": XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                )
    }

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, prefix):
    """
    Train multiple classifiers, evaluate them, save:
    - confusion matrix (best model)
    - ROC curves (all models)
    - bar plot of metrics
    - top-10 feature importance for Random Forest
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = get_classifiers()
    results = {}
    probas = {}

    # 1) Train all models
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        # some classifiers may not implement predict_proba
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            # fallback to decision_function
            y_score = clf.decision_function(X_test_scaled)
        probas[name] = y_score

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "y_pred": y_pred
        }

    # 2) Pick best model by accuracy
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_clf = classifiers[best_name]
    best_y_pred = results[best_name]["y_pred"]

    print(f"\n=== {prefix.upper()} – BEST MODEL: {best_name} ===")
    print(f"Accuracy: {results[best_name]['accuracy']:.3f}, "
          f"Precision: {results[best_name]['precision']:.3f}, "
          f"Recall: {results[best_name]['recall']:.3f}, "
          f"F1: {results[best_name]['f1']:.3f}")

    # 3) Confusion matrix (best model)
    cm = confusion_matrix(y_test, best_y_pred)
    # plt.figure(figsize=(5,4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.title(f"{prefix.upper()} – Confusion Matrix ({best_name})")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.tight_layout()
    # plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_confusion_matrix_{best_name}.png"),
    #             dpi=300, bbox_inches="tight")
    # plt.close()


    # 4) ROC curves for all classifiers
    cm = confusion_matrix(y_test, best_y_pred)
    labels = np.unique(y_test)

    # per-class metrics (same order as 'labels')
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, best_y_pred, labels=labels, zero_division=0
    )

    # Build annotation matrix: counts everywhere, P/R/F1 on diagonal cells
    annot = cm.astype(str).astype(object)
    for i, lab in enumerate(labels):
        # multi-line text inside diagonal cells
        annot[i, i] = (
            f"{cm[i, i]}\n"
            f"P={prec[i]:.2f}\n"
            f"R={rec[i]:.2f}\n"
            f"F1={f1[i]:.2f}"
        )

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"{prefix.upper()} – F1-scored Confusion Matrix ({best_name})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{prefix}_confusion_matrix_f1_{best_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


    # 5) Bar plot of metrics
    metrics_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
    metrics_df.plot(kind="bar", figsize=(10,5))
    plt.title(f"{prefix.upper()} – Classifier Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_classifier_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 6) Top-10 feature importance from Random Forest (if available)
    if "Random Forest" in classifiers:
        rf = classifiers["Random Forest"]
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            if len(importances) == len(feature_names):
                topk = min(10, len(importances))
                idx = np.argsort(importances)[-topk:][::-1]
                top_features = [feature_names[i] for i in idx]
                top_importances = importances[idx]

                plt.figure(figsize=(8,6))
                y_pos = np.arange(len(top_features))
                plt.barh(y_pos, top_importances[::-1])
                plt.yticks(y_pos, top_features[::-1])
                plt.xlabel("Feature importance")
                plt.title(f"{prefix.upper()} – Top {topk} Features (Random Forest)")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR,
                                         f"{prefix}_feature_importance_top{topk}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()

    return best_clf, scaler, results, classifiers


def run_cross_validation(k_folds=5):
    """
    Run participant-wise GroupKFold cross-validation for both IMU and HR.
    Uses the same preprocessing as run_full_pipeline(), but does not train final models.
    """
    # Build IMU data exactly as in run_full_pipeline
    imu_data, imu_feature_names = build_modality_data(imu_files, modality="imu")

    # Use aligned HR files (cubic / linear / nearest)
    hr_aligned_files = get_hr_aligned_files(method="cubic")  # or "linear" / "nearest"
    hr_data, hr_feature_names = build_hr_data_rowwise(hr_aligned_files)

    # Only participants that have both IMU and HR data
    common_pids = sorted(set(imu_data.keys()) & set(hr_data.keys()))
    print("\n[CV] Common participants with both IMU and HR data:", common_pids)

    if len(common_pids) < 2:
        print("[CV WARN] Not enough participants for cross-validation.")
        return

    # IMU CV
    cross_validate_modality(
        modality_data=imu_data,
        feature_names=imu_feature_names,
        pids=common_pids,
        prefix="imu_cv",
        k_folds=k_folds,
    )

    # HR CV
    cross_validate_modality(
        modality_data=hr_data,
        feature_names=hr_feature_names,
        pids=common_pids,
        prefix="hr_cv",
        k_folds=k_folds,
    )


# =============================================================================
# MAIN PIPELINE: TRAIN ON IMU + HR, 5 TRAIN / 3 TEST
# =============================================================================

def build_modality_data(files_dict, modality="imu"):
    """
    Load, preprocess, extract features, and segment into windows for each participant.
    Returns:
      data_by_pid: {pid: (X, y)}
      feature_names: list of feature names in window-level vectors
    """
    data_by_pid = {}
    feature_names = None

    for pid, path in files_dict.items():
        if not os.path.exists(path):
            print(f"[WARN] File not found for {modality} / {pid}: {path}")
            continue

        df = pd.read_csv(path)

        df = ensure_cybersickness_column(df)
        df = drop_non_feature_columns(df, label_col="Cybersickness")

        if modality == "imu":
            df = extract_imu_features(df)
        elif modality == "hr":
            df = extract_hr_features(df)
        else:
            raise ValueError("Unknown modality: " + modality)

        # After feature extraction, re-drop any non-feature columns that sneaked in
        df = drop_non_feature_columns(df, label_col="Cybersickness")

        # segment into windows (1 Hz, 10 seconds)
        X, y = segment_into_windows(df, label_col="Cybersickness", window_sec=1, sampling_hz=1)

        if X.size == 0:
            print(f"[WARN] No windows created for {modality} / {pid} (file: {path})")
            continue

        data_by_pid[pid] = (X, y)

        # get feature names only once (from numeric columns before segmentation)
        if feature_names is None:
            numeric_cols = df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number]).columns
            feature_names = list(numeric_cols)

    return data_by_pid, feature_names


def build_hr_data_rowwise(files_dict):
    """
    Load HR files, extract HR features, and return:
      data_by_pid: {pid: (X, y, bpm)} where each row is one time sample
      feature_names: list of HR feature names
    No windowing; classification is per-sample.
    """
    data_by_pid = {}
    feature_names = None

    for pid, path in files_dict.items():
        if not os.path.exists(path):
            print(f"[WARN] HR file not found for {pid}: {path}")
            continue

        df = pd.read_csv(path)

        # Ensure label name
        df = ensure_cybersickness_column(df)
        df = drop_non_feature_columns(df, label_col="Cybersickness")

        # HR / HRV-like features (uses 'bpm' column)
        df = extract_hr_features(df)
        df = drop_non_feature_columns(df, label_col="Cybersickness")

        # bpm series (cleaned) for fusion logic
        if "bpm" not in df.columns:
            print(f"[WARN] 'bpm' column missing for {pid} in {path}")
            continue

        bpm_arr = pd.to_numeric(df["bpm"], errors="coerce") \
                      .fillna(method="ffill") \
                      .fillna(method="bfill") \
                      .values

        # X: all numeric feature columns except label, y: label per row
        numeric_cols = df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])
        if numeric_cols.shape[1] == 0:
            print(f"[WARN] No numeric HR features for {pid} in {path}")
            continue

        X = numeric_cols.values
        y = df["Cybersickness"].values.astype(int)

        # ensure alignment of X, y, bpm
        min_len = min(len(X), len(y), len(bpm_arr))
        X = X[:min_len]
        y = y[:min_len]
        bpm_arr = bpm_arr[:min_len]

        data_by_pid[pid] = (X, y, bpm_arr)

        if feature_names is None:
            feature_names = list(numeric_cols.columns)

    return data_by_pid, feature_names

def fuse_predictions(imu_pred, hr_pred, bpm, bpm_threshold):
    """
    Fuse IMU and HR predictions using the rule:

    - If imu=1 and hr=1 -> 1
    - If imu=0 and hr=0 -> 0
    - If imu=1 and hr=0 -> 1 if bpm is high else 0
    - If imu=0 and hr=1 -> 0 if bpm is low else 1

    Here, "high"/"low" are defined w.r.t a scalar bpm_threshold.
    """
    fused = np.zeros_like(imu_pred)

    for i in range(len(fused)):
        if imu_pred[i] == 1 and hr_pred[i] == 1:
            fused[i] = 1
        else:
            fused[i] = 0
        # elif imu_pred[i] == 0 and hr_pred[i] == 0:
        #     fused[i] = 0
        # elif imu_pred[i] == 1 and hr_pred[i] == 0:
        #     # IMU says sick, HR says not; trust high BPM
        #     fused[i] = 1 if bpm[i] >= bpm_threshold else 0
        # else:  # imu_pred[i] == 0 and hr_pred[i] == 1
        #     # IMU says fine, HR says sick; trust low BPM to say fine
        #     fused[i] = 0 if bpm[i] <= bpm_threshold else 1

    return fused


def build_combined_data(imu_data, hr_data, imu_feature_names, hr_feature_names):
    """
    Build per-participant combined feature data:
      X_combined[t] = [X_imu[t] || X_hr[t]]
    Assumes IMU and HR windows are aligned per index for each pid.

    Returns:
      combined_data_by_pid: {pid: (X_combined, y_combined)}
      combined_feature_names: imu_feature_names + hr_feature_names
    """
    combined_data = {}
    combined_feature_names = imu_feature_names + hr_feature_names

    common_pids = sorted(set(imu_data.keys()) & set(hr_data.keys()))
    for pid in common_pids:
        X_imu, y_imu = imu_data[pid]
        X_hr, y_hr, _bpm = hr_data[pid]

        L = min(len(X_imu), len(X_hr), len(y_imu), len(y_hr))
        if L == 0:
            continue

        X_imu_use = X_imu[:L]
        X_hr_use  = X_hr[:L]
        # Labels should match; use IMU labels as reference
        y_use = y_imu[:L]

        X_comb = np.concatenate([X_imu_use, X_hr_use], axis=1)
        combined_data[pid] = (X_comb, y_use)

    return combined_data, combined_feature_names


def run_full_pipeline():
    # 1) Build IMU and HR datasets per participant
    imu_data, imu_feature_names = build_modality_data(imu_files, modality="imu")
    # hr_data, hr_feature_names = build_modality_data(hr_files, modality="hr")
    # hr_data, hr_feature_names = build_hr_data_rowwise(hr_files)
    # Choose which interpolation you want to use: 'cubic', 'linear', or 'nearest'
    hr_aligned_files = get_hr_aligned_files(method="cubic")  # or "linear" / "nearest"
    hr_data, hr_feature_names = build_hr_data_rowwise(hr_aligned_files)

    # Use only participants that have both IMU and HR data
    common_pids = sorted(set(imu_data.keys()) & set(hr_data.keys()))
    print("\nCommon participants with both IMU and HR data:", common_pids)

    if len(common_pids) < 8:
        print("[WARN] Fewer than 8 common participants; 5/3 split may not be exact.")

    # 2) 5 train / rest test (on common participants)
    train_pids = common_pids[:7]
    test_pids  = common_pids[7:]

    print("Train participants:", train_pids)
    print("Test participants: ", test_pids)

    # 3) Build train/test arrays for IMU
    X_train_imu = np.concatenate([imu_data[pid][0] for pid in train_pids], axis=0)
    y_train_imu = np.concatenate([imu_data[pid][1] for pid in train_pids], axis=0)
    X_test_imu  = np.concatenate([imu_data[pid][0] for pid in test_pids], axis=0)
    y_test_imu  = np.concatenate([imu_data[pid][1] for pid in test_pids], axis=0)

    # 4) Build train/test arrays for HR (and bpm)
    X_train_hr = np.concatenate([hr_data[pid][0] for pid in train_pids], axis=0)
    y_train_hr = np.concatenate([hr_data[pid][1] for pid in train_pids], axis=0)
    bpm_train  = np.concatenate([hr_data[pid][2] for pid in train_pids], axis=0)

    X_test_hr  = np.concatenate([hr_data[pid][0] for pid in test_pids], axis=0)
    y_test_hr  = np.concatenate([hr_data[pid][1] for pid in test_pids], axis=0)
    bpm_test   = np.concatenate([hr_data[pid][2] for pid in test_pids], axis=0)

    # --- Build COMBINED (IMU + HR) feature data per participant ---
    combined_data, combined_feature_names = build_combined_data(
        imu_data, hr_data, imu_feature_names, hr_feature_names
    )

    # Train/test arrays for combined features
    X_train_combined = np.concatenate([combined_data[pid][0] for pid in train_pids], axis=0)
    y_train_combined = np.concatenate([combined_data[pid][1] for pid in train_pids], axis=0)
    X_test_combined  = np.concatenate([combined_data[pid][0] for pid in test_pids], axis=0)
    y_test_combined  = np.concatenate([combined_data[pid][1] for pid in test_pids], axis=0)

    # 5) Train + evaluate IMU classifiers
    imu_best_model, imu_scaler, imu_results, imu_classifiers = train_and_evaluate(
        X_train_imu, y_train_imu, X_test_imu, y_test_imu,
        imu_feature_names, prefix="imu"
    )

    # 6) Train + evaluate HR classifiers
    hr_best_model, hr_scaler, hr_results, hr_classifiers = train_and_evaluate(
        X_train_hr, y_train_hr, X_test_hr, y_test_hr,
        hr_feature_names, prefix="hr"
    )

    # 6b) Train + evaluate COMBINED (IMU + HR) classifiers
    combined_best_model, combined_scaler, combined_results, combined_classifiers = train_and_evaluate(
        X_train_combined, y_train_combined,
        X_test_combined, y_test_combined,
        combined_feature_names, prefix="combined"
    )

    # Save ALL combined (IMU+HR) classifiers for later comparison with Transformer
    save_all_combined_classifiers(
        classifiers=combined_classifiers,
        scaler=combined_scaler,
        feature_names=combined_feature_names,
        train_pids=train_pids,
        test_pids=test_pids,
        results=combined_results,
    )


    # 7) FUSION EVALUATION (IMU + HR + bpm rule)

    # Threshold for "high" vs "low" bpm: use median of training bpm
    bpm_threshold = np.median(bpm_train)
    print(f"\nUsing bpm_threshold = {bpm_threshold:.2f} for fusion logic.")

    # Predictions on test sets with best models
    X_test_imu_scaled = imu_scaler.transform(X_test_imu)
    imu_pred_test = imu_best_model.predict(X_test_imu_scaled)

    X_test_hr_scaled = hr_scaler.transform(X_test_hr)
    hr_pred_test = hr_best_model.predict(X_test_hr_scaled)

    # We treat IMU predictions as the reference timeline
    L = len(imu_pred_test)

    # Sanity: make sure HR predictions and labels have the same length as IMU.
    # If not, we can truncate or warn; for now we truncate HR/y to L if needed.
    if len(hr_pred_test) != L:
        print(f"[WARN] hr_pred_test length {len(hr_pred_test)} != imu_pred_test length {L}, truncating to min.")
        L = min(L, len(hr_pred_test))
        imu_pred_test = imu_pred_test[:L]
        hr_pred_test = hr_pred_test[:L]

    if len(y_test_hr) != L:
        print(f"[WARN] y_test_hr length {len(y_test_hr)} != imu_pred_test length {L}, truncating to min.")
        L = min(L, len(y_test_hr))
        imu_pred_test = imu_pred_test[:L]
        hr_pred_test = hr_pred_test[:L]

    # Now imu_pred_test, hr_pred_test, y_test_hr all length L
    y_true_fused = y_test_hr[:L]

    # --- Cubic spline interpolation for bpm to length L ---
    # Original bpm_test has its own index 0..N-1
    N_bpm = len(bpm_test)
    x_old = np.arange(N_bpm)

    # If bpm length already matches L, no need to interpolate
    if N_bpm == L:
        bpm_test_use = bpm_test.copy()
    else:
        # New index spanning same range but with L points
        x_new = np.linspace(0, N_bpm - 1, L)
        cs = CubicSpline(x_old, bpm_test)
        bpm_test_use = cs(x_new)

    # --- Apply your fusion rule ---
    fused_pred = fuse_predictions(imu_pred_test, hr_pred_test, bpm_test_use, bpm_threshold)

    fused_acc  = accuracy_score(y_true_fused, fused_pred)
    fused_prec = precision_score(y_true_fused, fused_pred, zero_division=0)
    fused_rec  = recall_score(y_true_fused, fused_pred, zero_division=0)
    fused_f1   = f1_score(y_true_fused, fused_pred, zero_division=0)

    print(f"\n=== FUSED (IMU + HR + bpm rule) PERFORMANCE ===")
    print(f"Accuracy: {fused_acc:.3f}, Precision: {fused_prec:.3f}, "
          f"Recall: {fused_rec:.3f}, F1: {fused_f1:.3f}")


    # -------------------------------------------------------------------------
    # 8) 4-way comparison: IMU-only, HR-only, Combined, Fused (bpm rule)
    # -------------------------------------------------------------------------
    # Pick best classifier by accuracy in each modality
    best_imu_name = max(imu_results, key=lambda n: imu_results[n]["accuracy"])
    best_hr_name = max(hr_results, key=lambda n: hr_results[n]["accuracy"])
    best_combined_name = max(combined_results, key=lambda n: combined_results[n]["accuracy"])

    # ------------------------------------------------
    # SAVE BEST CLASSICAL MODELS FOR LATER COMPARISON
    # ------------------------------------------------
    save_best_ml_model(
        mode="imu",
        clf_name=best_imu_name,
        model=imu_best_model,
        scaler=imu_scaler,
        metrics=imu_results[best_imu_name],
    )

    save_best_ml_model(
        mode="hr",
        clf_name=best_hr_name,
        model=hr_best_model,
        scaler=hr_scaler,
        metrics=hr_results[best_hr_name],
    )

    save_best_ml_model(
        mode="combined",
        clf_name=best_combined_name,
        model=combined_best_model,
        scaler=combined_scaler,
        metrics=combined_results[best_combined_name],
    )

    comp = {
        "IMU-only": {
            "accuracy":  imu_results[best_imu_name]["accuracy"],
            "precision": imu_results[best_imu_name]["precision"],
            "recall":    imu_results[best_imu_name]["recall"],
            "f1":        imu_results[best_imu_name]["f1"],
        },
        "HR-only": {
            "accuracy":  hr_results[best_hr_name]["accuracy"],
            "precision": hr_results[best_hr_name]["precision"],
            "recall":    hr_results[best_hr_name]["recall"],
            "f1":        hr_results[best_hr_name]["f1"],
        },
        "Combined (IMU+HR)": {
            "accuracy":  combined_results[best_combined_name]["accuracy"],
            "precision": combined_results[best_combined_name]["precision"],
            "recall":    combined_results[best_combined_name]["recall"],
            "f1":        combined_results[best_combined_name]["f1"],
        },
        "Fused (bpm rule)": {
            "accuracy":  fused_acc,
            "precision": fused_prec,
            "recall":    fused_rec,
            "f1":        fused_f1,
        },
    }

    comp_df = pd.DataFrame(comp).T[["accuracy", "precision", "recall", "f1"]]
    plt.figure(figsize=(8, 5))
    comp_df.plot(kind="bar")
    plt.ylabel("Score")
    plt.title("IMU vs HR vs Combined vs Fused (bpm rule)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "comparison_imu_hr_combined_fused.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close()

    # Optional: F1-scored confusion matrix for fused predictions
    cm_fused = confusion_matrix(y_true_fused, fused_pred)
    labels = np.unique(y_true_fused)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true_fused, fused_pred, labels=labels, zero_division=0
    )

    annot = cm_fused.astype(str).astype(object)
    for i, lab in enumerate(labels):
        annot[i, i] = (
            f"{cm_fused[i, i]}\n"
            f"P={prec_c[i]:.2f}\n"
            f"R={rec_c[i]:.2f}\n"
            f"F1={f1_c[i]:.2f}"
        )

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm_fused,
        annot=annot,
        fmt="",
        cmap="Purples",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("FUSED – F1-scored Confusion Matrix (IMU + HR + bpm rule)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "fused_confusion_matrix_f1.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "imu": {
            "best_model": imu_best_model,
            "scaler": imu_scaler,
            "results": imu_results,
            "feature_names": imu_feature_names,
        },
        "hr": {
            "best_model": hr_best_model,
            "scaler": hr_scaler,
            "results": hr_results,
            "feature_names": hr_feature_names,
        },
    }


def run_cross_validation_with_fusion(k_folds=5, interp_method="cubic"):
    """
    Participant-wise K-fold CV for:
      - IMU-only
      - HR-only
      - Fused (IMU + HR + bpm rule)

    Uses aligned HR files (cubic/linear/nearest).
    """
    # Build IMU data
    imu_data, imu_feature_names = build_modality_data(imu_files, modality="imu")

    # Build HR data from aligned HR CSVs
    hr_aligned_files = get_hr_aligned_files(method=interp_method)
    hr_data, hr_feature_names = build_hr_data_rowwise(hr_aligned_files)

    # Common participants
    common_pids = sorted(set(imu_data.keys()) & set(hr_data.keys()))
    print("\n[CV] Common participants with both IMU and HR data:", common_pids)

    n_participants = len(common_pids)
    if n_participants < 2:
        print("[CV WARN] Not enough participants for cross-validation.")
        return

    n_splits = min(k_folds, n_participants)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)

    # Accumulators for metrics
    imu_metrics_all = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    hr_metrics_all  = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    fused_metrics_all = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(common_pids), start=1):
        train_pids = [common_pids[i] for i in train_idx]
        test_pids  = [common_pids[i] for i in test_idx]

        print(f"\n[CV] Fold {fold_idx}/{n_splits}")
        print("  Train participants:", train_pids)
        print("  Test participants: ", test_pids)

        # ----- Build train/test arrays for IMU -----
        X_train_imu = np.concatenate([imu_data[pid][0] for pid in train_pids], axis=0)
        y_train_imu = np.concatenate([imu_data[pid][1] for pid in train_pids], axis=0)
        X_test_imu  = np.concatenate([imu_data[pid][0] for pid in test_pids], axis=0)
        y_test_imu  = np.concatenate([imu_data[pid][1] for pid in test_pids], axis=0)

        # ----- Build train/test arrays for HR (and bpm) -----
        X_train_hr = np.concatenate([hr_data[pid][0] for pid in train_pids], axis=0)
        y_train_hr = np.concatenate([hr_data[pid][1] for pid in train_pids], axis=0)
        bpm_train  = np.concatenate([hr_data[pid][2] for pid in train_pids], axis=0)

        X_test_hr  = np.concatenate([hr_data[pid][0] for pid in test_pids], axis=0)
        y_test_hr  = np.concatenate([hr_data[pid][1] for pid in test_pids], axis=0)
        bpm_test   = np.concatenate([hr_data[pid][2] for pid in test_pids], axis=0)

        # ----- Train best IMU model for this fold -----
        imu_best_name, imu_best_clf, imu_scaler_fold, imu_pred_test, imu_best_metrics = \
            train_best_for_fold(X_train_imu, y_train_imu, X_test_imu, y_test_imu)

        print(f"  [IMU] Best model: {imu_best_name} | "
              f"Acc={imu_best_metrics['accuracy']:.3f}, "
              f"P={imu_best_metrics['precision']:.3f}, "
              f"R={imu_best_metrics['recall']:.3f}, "
              f"F1={imu_best_metrics['f1']:.3f}")

        for k in imu_metrics_all:
            imu_metrics_all[k].append(imu_best_metrics[k])

        # ----- Train best HR model for this fold -----
        hr_best_name, hr_best_clf, hr_scaler_fold, hr_pred_test, hr_best_metrics = \
            train_best_for_fold(X_train_hr, y_train_hr, X_test_hr, y_test_hr)

        print(f"  [HR]  Best model: {hr_best_name} | "
              f"Acc={hr_best_metrics['accuracy']:.3f}, "
              f"P={hr_best_metrics['precision']:.3f}, "
              f"R={hr_best_metrics['recall']:.3f}, "
              f"F1={hr_best_metrics['f1']:.3f}")

        for k in hr_metrics_all:
            hr_metrics_all[k].append(hr_best_metrics[k])

        # ----- FUSION on this fold -----
        # IMU / HR preds already correspond sample-wise by construction:
        # we concatenated participants in the same order.

        # bpm threshold from TRAIN bpm
        bpm_threshold = np.median(bpm_train)

        # Ensure lengths match (they should, but be safe)
        L = min(len(imu_pred_test), len(hr_pred_test), len(bpm_test), len(y_test_hr))
        imu_pred_use = imu_pred_test[:L]
        hr_pred_use  = hr_pred_test[:L]
        bpm_use      = bpm_test[:L]
        y_true_fused = y_test_hr[:L]

        fused_pred = fuse_predictions(imu_pred_use, hr_pred_use, bpm_use, bpm_threshold)

        fused_acc  = accuracy_score(y_true_fused, fused_pred)
        fused_prec = precision_score(y_true_fused, fused_pred, zero_division=0)
        fused_rec  = recall_score(y_true_fused, fused_pred, zero_division=0)
        fused_f1   = f1_score(y_true_fused, fused_pred, zero_division=0)

        print(f"  [FUSED] Acc={fused_acc:.3f}, P={fused_prec:.3f}, "
              f"R={fused_rec:.3f}, F1={fused_f1:.3f}")

        fused_metrics_all["accuracy"].append(fused_acc)
        fused_metrics_all["precision"].append(fused_prec)
        fused_metrics_all["recall"].append(fused_rec)
        fused_metrics_all["f1"].append(fused_f1)

    # ----- Summary across folds -----
    def print_summary(tag, m):
        print(f"\n=== {tag} – {n_splits}-fold CV summary ===")
        for key in ["accuracy", "precision", "recall", "f1"]:
            vals = np.array(m[key])
            print(f"{key.capitalize():>9}: {vals.mean():.3f} ± {vals.std():.3f}")

    print_summary("IMU-only", imu_metrics_all)
    print_summary("HR-only", hr_metrics_all)
    print_summary("FUSED (IMU+HR+bpm)", fused_metrics_all)

# =============================================================================
# INFERENCE HELPER: GIVEN NEW IMU + HR DATA, OUTPUT 0/1 FOR EACH
# =============================================================================

def predict_from_raw(imu_df, hr_df, imu_model, imu_scaler, hr_model, hr_scaler):
    """
    Given raw IMU and HR dataframes with 'Cybersickness' column (can be dummy),
    return predicted labels (0/1) per 10-second window for each modality.
    """
    # IMU
    imu_df = ensure_cybersickness_column(imu_df)
    imu_df = drop_non_feature_columns(imu_df, "Cybersickness")
    imu_df = extract_imu_features(imu_df)
    imu_df = drop_non_feature_columns(imu_df, "Cybersickness")

    X_imu, _ = segment_into_windows(imu_df, label_col="Cybersickness", window_sec=10, sampling_hz=1)
    X_imu_scaled = imu_scaler.transform(X_imu)
    imu_preds = imu_model.predict(X_imu_scaled)

    # # HR
    # hr_df = ensure_cybersickness_column(hr_df)
    # hr_df = drop_non_feature_columns(hr_df, "Cybersickness")
    # hr_df = extract_hr_features(hr_df)
    # hr_df = drop_non_feature_columns(hr_df, "Cybersickness")

    # X_hr, _ = segment_into_windows(hr_df, label_col="Cybersickness", window_sec=10, sampling_hz=1)
    # X_hr_scaled = hr_scaler.transform(X_hr)
    # hr_preds = hr_model.predict(X_hr_scaled)

    # HR – row-wise classification (no windowing)
    hr_df = ensure_cybersickness_column(hr_df)
    hr_df = drop_non_feature_columns(hr_df, "Cybersickness")
    hr_df = extract_hr_features(hr_df)
    hr_df = drop_non_feature_columns(hr_df, "Cybersickness")

    numeric_cols_hr = hr_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])
    X_hr = numeric_cols_hr.values
    X_hr_scaled = hr_scaler.transform(X_hr)
    hr_preds = hr_model.predict(X_hr_scaled)   # 0/1 per sample (per second)


    return imu_preds, hr_preds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # create_all_interpolated_hr_csvs()
    # 1) Cross-validation first
    run_cross_validation_with_fusion(k_folds=5, interp_method="cubic")

    # # 2) Then full train/test + 4-way comparison & plots
    models = run_full_pipeline()
