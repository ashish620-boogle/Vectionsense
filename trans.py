import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

HR_INTERP_METHOD = "cubic"   # "cubic", "linear", "nearest"

# Window / sequence settings (you currently use 1-second windows)
WINDOW_SEC = 1
SAMPLING_HZ = 1
SEQ_LEN = WINDOW_SEC * SAMPLING_HZ

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
PLOTS_DIR = "plots_transformer"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use("default")
plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 1.3,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.weight": "bold",
    "axes.grid": False,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
})


# =============================================================================
# FILE PATHS
# =============================================================================

imu_files = {
    "arpit":      "RC Cybersickness detection - rc_arpit_combined_imu.csv",
    "himanshu":   "RC Cybersickness detection - rc_himanshu_combined_imu.csv",
    "jatin":      "RC Cybersickness detection - rc_jatin_combined_imu.csv",
    "mihir":      "RC Cybersickness detection - rc_mihir_combined_imu.csv",
    "riya":       "RC Cybersickness detection - rc_riya_combined_imu.csv",
    "shreyadeb":  "RC Cybersickness detection - rc_shreyadeb_combined_imu.csv",
    "shubham":    "RC Cybersickness detection - rc_subham_combined_imu.csv",
    "vaishnavi":  "RC Cybersickness detection - rc_vaishnavi_combined_imu.csv",
    "rounak":     "RC Cybersickness detection - rc_rounak_combined_imu.csv",
    "shreya":     "RC Cybersickness detection - rc_shreya_combined_imu.csv",
}

def get_hr_aligned_files(method="cubic"):
    """
    Return dict {pid: path} for interpolated HRV CSVs aligned to IMU length.
    (You already created these via your interpolation script.)
    """
    aligned_root = os.path.join("hr_aligned", method)
    hr_aligned_files = {}
    for pid in imu_files.keys():
        path = os.path.join(aligned_root, f"{pid}_hr_aligned_{method}.csv")
        if os.path.exists(path):
            hr_aligned_files[pid] = path
    return hr_aligned_files

# =============================================================================
# HELPERS
# =============================================================================

def ensure_cybersickness_column(df):
    """Ensure label column is exactly named 'Cybersickness'."""
    if "Cybersickness" in df.columns:
        return df
    cand = [c for c in df.columns if "cyber" in c.lower()]
    if not cand:
        raise ValueError(f"No Cybersickness-like column found in columns: {df.columns.tolist()}")
    return df.rename(columns={cand[0]: "Cybersickness"})


def drop_non_feature_columns(df, label_col="Cybersickness"):
    """
    Drop obvious non-feature columns (comments, filenames, time, etc).
    Keep label_col.
    """
    cols_to_drop = []
    for c in df.columns:
        cl = c.lower()
        if c == label_col:
            continue
        if "comment" in cl or "file" in cl or "participant" in cl or "name" in cl:
            cols_to_drop.append(c)
        if cl == "time" or cl == "seconds_elapsed":
            cols_to_drop.append(c)

    if cols_to_drop:
        df = df.drop(columns=list(set(cols_to_drop)))
    return df


def align_hr_to_imu_length(hr_path, imu_path):
    """
    Reads hr.csv and imu.csv, performs cubic spline interpolation on BPM
    to match the IMU row count.
    Returns a new DataFrame with columns: ['bpm', 'Cybersickness']
    """
    if not os.path.exists(hr_path) or not os.path.exists(imu_path):
        print(f"[ALIGN] Missing HR/IMU file: {hr_path} or {imu_path}")
        return None

    try:
        hr_df = pd.read_csv(hr_path)
        imu_df = pd.read_csv(imu_path)
    except Exception as e:
        print(f"[ALIGN] Could not read CSV: {e}")
        return None

    # Ensure columns
    if "bpm" not in hr_df.columns:
        raise ValueError(f"'bpm' not found in HR file: {hr_path}")

    if "Cybersickness" not in imu_df.columns:
        imu_df = ensure_cybersickness_column(imu_df)

    # Target length
    target_len = len(imu_df)

    # Existing HR samples
    hr_bpm = hr_df["bpm"].astype(float).values
    n_old = len(hr_bpm)

    if n_old < 2:
        print(f"[ALIGN] HR file too small ({n_old} rows): {hr_path}")
        return None

    # Index positions
    x_old = np.arange(n_old)
    x_new = np.linspace(0, n_old - 1, target_len)

    # Cubic spline interpolation
    try:
        cs = CubicSpline(x_old, hr_bpm)
        bpm_new = cs(x_new)
    except Exception as e:
        print(f"[ALIGN] Spline failed on {hr_path}: {e}")
        return None

    # Output aligned HR dataframe
    aligned_df = pd.DataFrame({
        "bpm": bpm_new,
        "Cybersickness": imu_df["Cybersickness"].values
    })

    return aligned_df


def extract_imu_features(df):
    """
    From raw IMU (head + controllers), derive velocity, acceleration,
    jerk, sway variance, trajectory smoothness, movement energy.
    """
    df = df.copy()
    imu_cols = [c for c in df.columns if any(axis in c.lower() for axis in ["x", "y", "z"])]

    # velocity
    for c in imu_cols:
        df[f"vel_{c}"] = np.gradient(df[c].values)

    # acceleration
    for c in imu_cols:
        df[f"acc_{c}"] = np.gradient(df[f"vel_{c}"].values)

    # jerk
    for c in imu_cols:
        df[f"jerk_{c}"] = np.gradient(df[f"acc_{c}"].values)

    # sway
    df["sway_var"] = df[imu_cols].select_dtypes(include=[np.number]).var(axis=1)

    # smoothness
    jerk_cols = [c for c in df.columns if c.startswith("jerk_")]
    if jerk_cols:
        df["trajectory_smoothness"] = np.linalg.norm(df[jerk_cols].values, axis=1)
    else:
        df["trajectory_smoothness"] = 0.0

    # movement energy
    acc_cols = [c for c in df.columns if c.startswith("acc_")]
    if acc_cols:
        df["movement_energy"] = (df[acc_cols].values ** 2).sum(axis=1)
    else:
        df["movement_energy"] = 0.0

    # cleanup accidental label-derivatives
    bad_cs_cols = ["vel_Cybersickness", "acc_Cybersickness", "jerk_Cybersickness"]
    present_bad = [c for c in bad_cs_cols if c in df.columns]
    if present_bad:
        df = df.drop(columns=present_bad)

    return df


def extract_hr_features(df):
    """
    Basic HRV-like features from HR data.
    Assumes 'bpm' column exists.
    """
    df = df.copy()
    if "bpm" not in df.columns:
        raise ValueError(f"'bpm' not found in columns: {df.columns.tolist()}")

    df["hr"] = pd.to_numeric(df["bpm"], errors="coerce") \
                  .fillna(method="ffill").fillna(method="bfill")
    df["hr_diff"] = df["hr"].diff().fillna(0.0)
    df["hr_sq"] = df["hr"] ** 2

    hr_safe = df["hr"].replace(0, np.nan)
    df["rr"] = (60.0 / hr_safe).fillna(method="ffill").fillna(method="bfill")
    df["rr_diff"] = df["rr"].diff().fillna(0.0)
    return df


def get_predictions_and_labels(model, loader):
    """
    Run model on a loader and return concatenated (y_true, y_pred).
    """
    model.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    return all_y, all_pred

def plot_confusion_matrix_compact(cm, class_labels, out_path, title=None):
    """
    Compact confusion matrix:
      - Blues colormap
      - Bold labels
      - White grid between cells
      - Larger annotation fonts
      - No title by default
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # Ticks & labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=25, ha="right",
                       fontweight="bold", fontsize=15)
    ax.set_yticklabels(class_labels, fontweight="bold", fontsize=15)

    ax.set_xlabel("Predicted", fontweight="bold", fontsize=16)
    ax.set_ylabel("True", fontweight="bold", fontsize=16)
    # ⛔ no title any more

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontweight="bold", fontsize=15)

    # Grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(class_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with bold values, larger font
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val:d}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=18,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_loocv_participantwise(loocv_df, out_path_png, out_path_pdf=None):
    """
    Participant-wise grouped bar plot for LOOCV metrics,
    styled similar to your WPM participant plot.
    
    loocv_df: DataFrame with index = participant IDs (e.g., "arpit"),
              columns = ["accuracy", "precision", "recall", "f1"].
    """

    # -------------------------
    # Styling (aligned with WPM fig)
    # -------------------------
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    # 4 metrics instead of 3 scenarios → extend colors & hatches
    METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]
    METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-score"]

    COLORS  = ["#4e79a7", "#e15759", "#7c48ff", "#59a14f"]
    HATCHES = ["////", "\\\\\\\\", "....", "xx"]

    # Participants as P01, P02, ... (or use raw ids if you prefer)
    pids = list(loocv_df.index)
    n_participants = len(pids)
    x = np.arange(n_participants)

    # Labels to show on x-axis: "P01", "P02", ...
    x_labels = [f"P{idx+1:02d}" for idx in range(n_participants)]

    width = 0.18   # thinner bars, close spacing (like your WPM code)

    fig, ax = plt.subplots(figsize=(12, 3))

    for i, (metric, mlab) in enumerate(zip(METRIC_NAMES, METRIC_LABELS)):
        if metric not in loocv_df.columns:
            continue

        values = loocv_df[metric].values
        # Center metrics around zero with offset (i - (n_metrics-1)/2)*width
        offset = (i - (len(METRIC_NAMES)-1)/2) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=mlab,
            color=COLORS[i],
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCHES[i],
        )

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Keep y-axis within [0,1] since these are metric scores
    ax.set_ylim(0, 1.05)

    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", ncols=len(METRIC_NAMES), frameon=True, facecolor="white")

    fig.tight_layout()
    fig.savefig(out_path_png, bbox_inches="tight", dpi=300)
    if out_path_pdf is not None:
        fig.savefig(out_path_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

# =============================================================================
# BUILD SEQUENCES
# =============================================================================

def build_sequences_for_pid(pid, imu_path, hr_path_aligned=None,
                            window_sec=10, sampling_hz=1, mode="both"):
    """
    Build windowed sequences for a single participant.

    mode:
        "imu"  -> only IMU features
        "hr"   -> only HR features (requires hr_path_aligned)
        "both" -> concatenated IMU+HR features

    Returns:
        seq_list: list of [T, D]
        labels : np.array of int
        feature_names: list of D column names
    """
    if not os.path.exists(imu_path):
        print(f"[WARN] IMU file missing for {pid}: {imu_path}")
        return [], np.array([]), None

    imu_df = pd.read_csv(imu_path)
    imu_df = ensure_cybersickness_column(imu_df)
    imu_df = drop_non_feature_columns(imu_df, "Cybersickness")
    imu_df = extract_imu_features(imu_df)
    imu_df = drop_non_feature_columns(imu_df, "Cybersickness")

    label_series = None
    X_all = None
    feature_names = None

    if hr_path_aligned is not None and os.path.exists(hr_path_aligned):
        # HR aligned to IMU length
        hr_df = pd.read_csv(hr_path_aligned)
        hr_df = ensure_cybersickness_column(hr_df)
        hr_df = drop_non_feature_columns(hr_df, "Cybersickness")
        hr_df = extract_hr_features(hr_df)
        hr_df = drop_non_feature_columns(hr_df, "Cybersickness")

        # Align lengths (should already be same if aligned properly)
        L = min(len(imu_df), len(hr_df))
        imu_df = imu_df.iloc[:L].reset_index(drop=True)
        hr_df  = hr_df.iloc[:L].reset_index(drop=True)

        label_series = imu_df["Cybersickness"].values

        X_imu_df = imu_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])
        X_hr_df  = hr_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])

        X_imu = X_imu_df.values
        X_hr  = X_hr_df.values

        imu_cols = list(X_imu_df.columns)
        hr_cols  = list(X_hr_df.columns)

        if mode == "both":
            X_all = np.concatenate([X_imu, X_hr], axis=1)
            feature_names = imu_cols + hr_cols
        elif mode == "imu":
            X_all = X_imu
            feature_names = imu_cols
        elif mode == "hr":
            X_all = X_hr
            feature_names = hr_cols
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        # IMU-only case (e.g. no HR aligned file)
        if mode == "hr":
            print(f"[WARN] No HR aligned file for {pid}, skipping in HR-only mode.")
            return [], np.array([]), None

        label_series = imu_df["Cybersickness"].values
        X_imu_df = imu_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])
        X_all = X_imu_df.values
        feature_names = list(X_imu_df.columns)

    # Windowing
    window_size = window_sec * sampling_hz
    rows = len(label_series)

    seq_list = []
    label_list = []

    for start in range(0, rows, window_size):
        end = start + window_size
        if end > rows:
            break

        X_chunk = X_all[start:end]
        y_chunk = label_series[start:end]

        if X_chunk.shape[0] != window_size:
            continue

        label = int(y_chunk.mean() >= 0.5)
        seq_list.append(X_chunk.astype(np.float32))
        label_list.append(label)

    return seq_list, np.array(label_list, dtype=np.int64), feature_names


def build_all_sequences(mode="both", method="cubic"):
    """
    Build sequences for all participants for a given mode:
      - "imu"
      - "hr"
      - "both"

    Returns:
        data_by_pid: {pid: (seq_list, labels)}
        feature_names: list of feature names
    """
    data_by_pid = {}
    global_feature_names = None

    hr_map = get_hr_aligned_files(method)

    for pid, imu_path in imu_files.items():
        hr_path = hr_map.get(pid, None)

        seq_list, labels, feat_names = build_sequences_for_pid(
            pid,
            imu_path,
            hr_path_aligned=hr_path,
            window_sec=WINDOW_SEC,
            sampling_hz=SAMPLING_HZ,
            mode=mode,
        )

        if len(seq_list) == 0:
            continue

        data_by_pid[pid] = (seq_list, labels)
        print(f"[INFO] {mode.upper()} – {pid}: {len(seq_list)} sequences")

        if global_feature_names is None and feat_names is not None:
            global_feature_names = feat_names
        elif feat_names is not None and feat_names != global_feature_names:
            print(f"[WARN] Feature name mismatch for {pid} in mode={mode}; "
                  f"using feature_names from first participant.")

    return data_by_pid, global_feature_names

# =============================================================================
# DATASET
# =============================================================================

class SeqDataset(Dataset):
    def __init__(self, seq_list, label_array):
        self.seq_list = seq_list
        self.labels = label_array

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.seq_list[idx]      # [T, D]
        y = self.labels[idx]
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)   # [B, T, D]
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2,
                 num_classes=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: [B, T, feature_dim]
        x = self.input_proj(x)      # [B, T, d_model]
        x = self.pos_enc(x)         # positional encoding
        x = self.encoder(x)         # [B, T, d_model]
        x = x.mean(dim=1)           # global average over time
        logits = self.cls_head(x)   # [B, num_classes]
        return logits

# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_y = []
    all_pred = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

        preds = logits.argmax(dim=1)
        all_y.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, zero_division=0)
    return avg_loss, acc, f1


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_y = []
    all_pred = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)

            preds = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    prec = precision_score(all_y, all_pred, zero_division=0)
    rec = recall_score(all_y, all_pred, zero_division=0)
    f1 = f1_score(all_y, all_pred, zero_division=0)
    return avg_loss, acc, prec, rec, f1


def transformer_feature_importance(model, test_ds, feature_names, topk=10, save_path=None):
    """
    Permutation-based feature importance for the Transformer.
    Uses real column names from feature_names.
    """
    model.eval()

    # Build full test arrays
    X_list = test_ds.seq_list
    y = test_ds.labels
    X = np.stack(X_list, axis=0)   # [N, T, D]
    N, T, D = X.shape

    # Baseline predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    baseline_f1 = f1_score(y, y_pred, zero_division=0)
    importances = np.zeros(D, dtype=np.float32)

    for j in range(D):
        X_perm = X.copy()
        flat = X_perm[:, :, j].reshape(N, T)
        # Shuffle along N for each time step t
        for t in range(T):
            np.random.shuffle(flat[:, t])
        X_perm[:, :, j] = flat

        with torch.no_grad():
            X_perm_tensor = torch.tensor(X_perm, dtype=torch.float32, device=device)
            logits_perm = model(X_perm_tensor)
            y_pred_perm = logits_perm.argmax(dim=1).cpu().numpy()

        f1_perm = f1_score(y, y_pred_perm, zero_division=0)
        importances[j] = baseline_f1 - f1_perm

    # Top-k
    topk = min(topk, D)
    idx_sorted = np.argsort(importances)[-topk:][::-1]
    top_importances = importances[idx_sorted]

    if feature_names is not None and len(feature_names) == D:
        feat_names_all = feature_names
    else:
        feat_names_all = [f"f{j}" for j in range(D)]

    top_names = [feat_names_all[j] for j in idx_sorted]

    # --- Plot with larger fonts ---
    plt.figure(figsize=(7, 5))
    y_pos = np.arange(len(top_names))

    plt.barh(y_pos, top_importances[::-1])
    plt.yticks(y_pos, top_names[::-1], fontsize=20)
    plt.xlabel("Permutation importance (ΔF1)", fontsize=22, fontweight="bold")
    # no title (to match your other figs)

    ax = plt.gca()
    ax.tick_params(axis="x", labelsize=20, labelrotation=45)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_longevity_experiment(
    model,
    scaler,
    expected_feature_names,
    window_sec=WINDOW_SEC,
    sampling_hz=SAMPLING_HZ,
    longevity_dir="."
):
    """
    Longevity experiment:
    - Looks for IMU+HR files named like:
        <pid><day>_combined_imu.csv   e.g., k1_combined_imu.csv, a1_combined_imu.csv
        <pid><day> hr.csv             e.g., k1 hr.csv, a1 hr.csv, k2 hr.csv
    - For each participant (pid) and each day:
        * build COMBINED (IMU+HR) sequences
        * scale with the already-trained 'scaler'
        * evaluate with the already-trained 'model'
    - Saves:
        * longevity_results.csv (participant, day, acc/prec/rec/f1)
        * longevity_curve_metrics_over_days.png (mean curves across participants)
    """

    # -------------------------------------------------
    # 1) Discover day-wise IMU and HR files
    # -------------------------------------------------
    imu_day_files = {}  # pid -> {day: path}
    hr_day_files = {}   # pid -> {day: path}

    for fname in os.listdir(longevity_dir):
        full_path = os.path.join(longevity_dir, fname)

        if not os.path.isfile(full_path):
            continue

        # IMU: strict pattern "<letters><digits>_combined_imu.csv"
        if fname.endswith("_combined_imu.csv"):
            base = fname[:-len("_combined_imu.csv")]  # e.g. "k1", "a2"
            if " " in base:
                continue
            if len(base) < 2 or (not base[-1].isdigit()) or (not base[-2].isalpha()):
                continue

            pid = "".join([c for c in base if c.isalpha()])
            day_str = "".join([c for c in base if c.isdigit()])
            day = int(day_str) if day_str else 1
            imu_day_files.setdefault(pid, {})[day] = full_path

        # HR: strict pattern "<letters><digits> hr.csv"
        elif fname.endswith(" hr.csv"):
            base = fname[:-len(" hr.csv")]  # e.g. "k1", "a1"
            if len(base) < 2 or (not base[-1].isdigit()) or (not base[-2].isalpha()):
                continue

            pid = "".join([c for c in base if c.isalpha()])
            day_str = "".join([c for c in base if c.isdigit()])
            day = int(day_str) if day_str else 1
            hr_day_files.setdefault(pid, {})[day] = full_path

    if not imu_day_files or not hr_day_files:
        print("[LONGEVITY] No longevity-style files found (e.g., k1_combined_imu.csv / k1 hr.csv). Skipping.")
        return

    # -------------------------------------------------
    # 2) Evaluate model day-wise for each participant
    # -------------------------------------------------
    rows = []
    criterion = nn.CrossEntropyLoss()

    for pid in sorted(imu_day_files.keys()):
        if pid not in hr_day_files:
            continue

        common_days = sorted(set(imu_day_files[pid].keys()) & set(hr_day_files[pid].keys()))
        if not common_days:
            continue

        print(f"[LONGEVITY] Participant '{pid}' has days: {common_days}")

        for day in common_days:
            imu_path = imu_day_files[pid][day]
            hr_raw_path = hr_day_files[pid][day]

            # --- Align HR to IMU using cubic spline ---
            aligned_hr_df = align_hr_to_imu_length(hr_raw_path, imu_path)

            if aligned_hr_df is None:
                print(f"[LONGEVITY] Failed HR alignment for {pid}, day {day}. Skipping.")
                continue

            # Save temporary aligned HR file (in-memory df → csv)
            temp_hr_path = os.path.join(PLOTS_DIR, f"temp_{pid}_day{day}_hr_aligned.csv")
            aligned_hr_df.to_csv(temp_hr_path, index=False)

            # Build COMBINED (IMU+HR) sequences for this (pid, day)
            seq_list, labels, feat_names = build_sequences_for_pid(
                pid=f"{pid}_day{day}",
                imu_path=imu_path,
                hr_path_aligned=temp_hr_path,
                window_sec=window_sec,
                sampling_hz=sampling_hz,
                mode="both",
            )

            if len(seq_list) == 0:
                print(f"[LONGEVITY] No sequences for {pid}, day {day}. Skipping.")
                continue

            # --- Align local features to the training feature set (expected_feature_names) ---
            if feat_names is None or expected_feature_names is None:
                print(f"[LONGEVITY] Missing feature names for {pid}, day {day}. Skipping.")
                continue

            exp_names = expected_feature_names              # list of training feature names
            name_to_idx = {n: i for i, n in enumerate(feat_names)}
            D_exp = len(exp_names)

            aligned_seq_list = []
            missing_count = 0

            for seq in seq_list:
                T, _ = seq.shape
                aligned = np.zeros((T, D_exp), dtype=np.float32)

                for j, fname in enumerate(exp_names):
                    if fname in name_to_idx:
                        aligned[:, j] = seq[:, name_to_idx[fname]]
                    else:
                        # feature not present in this day recording -> keep zeros
                        missing_count += 1

                aligned_seq_list.append(aligned)

            if missing_count > 0:
                print(
                    f"[LONGEVITY] Warning: {missing_count} feature slots missing "
                    f"for {pid}, day {day} (filled with zeros)."
                )

            # Apply SAME scaler used for the combined model
            scaled_seqs = [scaler.transform(seq) for seq in aligned_seq_list]
            ds = SeqDataset(scaled_seqs, labels)
            loader = DataLoader(ds, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn)

            # --- Model predictions for this (pid, day) ---
            y_true_day, y_pred_day = get_predictions_and_labels(model, loader)

            # Classification metrics (still stored in longevity_results.csv)
            acc  = accuracy_score(y_true_day, y_pred_day)
            prec = precision_score(y_true_day, y_pred_day, zero_division=0)
            rec  = recall_score(y_true_day, y_pred_day, zero_division=0)
            f1   = f1_score(y_true_day, y_pred_day, zero_division=0)

            # --- NEW: count of cybersickness detections (predicted 1) for this day ---
            detections = int((y_pred_day == 1).sum())

            # --- NEW: average HRV proxy from aligned HR (SDNN over RR intervals) ---
            bpm_arr = aligned_hr_df["bpm"].astype(float).values
            hr_safe = np.where(bpm_arr > 0, bpm_arr, np.nan)
            rr = 60.0 / hr_safe                  # RR intervals in seconds
            hrv_sdnn = float(np.nanstd(rr))      # HRV proxy (SDNN)

            rows.append({
                "participant": pid,
                "day": day,
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1":        f1,
                "detections": detections,   # NEW
                "hrv_sdnn":   hrv_sdnn,     # NEW
                "avg_hrv":   hrv_sdnn,     # NEW
            })

            print(
                f"[LONGEVITY] {pid} – Day {day}: "
                f"acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}, "
                f"detections={detections}, hrv_sdnn={hrv_sdnn:.4f}"
            )


    if not rows:
        print("[LONGEVITY] No valid (participant, day) sequences. Skipping plots.")
        return

    longevity_df = pd.DataFrame(rows)
    longevity_csv_path = os.path.join(PLOTS_DIR, "longevity_results.csv")
    longevity_df.to_csv(longevity_csv_path, index=False)
    print(f"[LONGEVITY] Saved per-day results to {longevity_csv_path}")

    # -------------------------------------------------
    # Cybersickness count + average HRV per participant
    # -------------------------------------------------
    if {"detections", "avg_hrv"}.issubset(longevity_df.columns):

        # Aggregate over days: total cybersickness, mean HRV per participant
        agg_df = longevity_df.groupby("participant").agg(
            total_detections=("detections", "sum"),
            mean_hrv=("avg_hrv", "mean")
        ).reset_index()

        participants = agg_df["participant"].tolist()
        x = np.arange(len(participants))

        counts = agg_df["total_detections"].values
        mean_hrv = agg_df["mean_hrv"].values

        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        })

        fig, ax1 = plt.subplots(figsize=(7, 4))

        # --- Bars: # cybersickness detections ---
        bar_width = 0.6
        bars = ax1.bar(
            x,
            counts,
            width=bar_width,
            color="#4e79a7",
            edgecolor="black",
            label="# cybersickness detections",
        )
        ax1.set_xlabel("Participant")
        ax1.set_ylabel("# cybersickness detections")
        ax1.set_xticks(x)
        ax1.set_xticklabels(participants, rotation=0)

        # --- Second axis: line for avg HR/HRV ---
        ax2 = ax1.twinx()
        ax2.plot(
            x,
            mean_hrv,
            marker="o",
            linewidth=2.0,
            color="#e15759",
            label="Avg HRV / BPM",
        )
        ax2.set_ylabel("Average HRV / heart rate")

        # --- Combine legends from both axes ---
        lines, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            lines.extend(h)
            labels.extend(l)

        leg = ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white")
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

        fig.tight_layout()
        out_path = os.path.join(PLOTS_DIR, "participant_cybersickness_vs_hrv.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[LONGEVITY] Saved participant cybersickness/HRV plot to {out_path}")
    else:
        print("[LONGEVITY] 'detections' or 'avg_hrv' missing in longevity_df; "
              "skipping participant summary plot.")




# =============================================================================
# LEAVE-ONE-OUT CROSS VALIDATION (LOOCV) FOR COMBINED MODEL
# =============================================================================

def run_loocv_combined(data_both, feature_names):
    """
    Performs Leave-One-Out Cross Validation (LOOCV) over participants
    for the COMBINED (IMU + HR) dataset.
    For each participant:
      - train on all others
      - test on the left-out participant
      - record accuracy, precision, recall, f1
    Produces:
      - CSV of results
      - 4 bar plots: accuracy, precision, recall, f1 vs participant
    """

    participant_ids = sorted(list(data_both.keys()))
    results = []

    for test_pid in participant_ids:
        print(f"\n========== LOOCV: Leaving out {test_pid} as test ==========")

        train_pids = [p for p in participant_ids if p != test_pid]
        test_pids  = [test_pid]

        # Train transformer on combined model
        res = train_transformer_for_dataset(
            data_by_pid=data_both,
            feature_names=feature_names,
            train_pids=train_pids,
            test_pids=test_pids,
            tag=f"combined_LOOCV_{test_pid}"
        )

        # Store metrics
        results.append({
            "participant": test_pid,
            "accuracy":  res["metrics"]["accuracy"],
            "precision": res["metrics"]["precision"],
            "recall":    res["metrics"]["recall"],
            "f1":        res["metrics"]["f1"],
        })

    # Convert to DataFrame
    loocv_df = pd.DataFrame(results)
    loocv_df.to_csv(os.path.join(PLOTS_DIR, "LOOCV_combined_results.csv"), index=False)
    print("\n[LOOCV] Saved LOOCV results to LOOCV_combined_results.csv")

    # === PLOTTING ===
    # loocv_df: index = pid, columns = ["accuracy", "precision", "recall", "f1"]
    plot_loocv_participantwise(
        loocv_df,
        out_path_png=os.path.join(PLOTS_DIR, "loocv_participantwise_metrics.png"),
        out_path_pdf=os.path.join(PLOTS_DIR, "loocv_participantwise_metrics.pdf"),
    )

    print("[LOOCV] Saved LOOCV plots in:", PLOTS_DIR)


# =============================================================================
# TRAIN ONE TRANSFORMER FOR A GIVEN MODE (IMU / HR / BOTH)
# =============================================================================

def train_transformer_for_dataset(data_by_pid, feature_names, train_pids, test_pids, tag):
    """
    Train + evaluate a Transformer for a given feature mode.
    Returns:
      {
        "metrics": {acc, precision, recall, f1},
        "y_true": np.array,
        "y_pred": np.array
      }
    Also saves confusion matrix and feature-importance plot.
    """
    # 1) Build global train/test sequence lists
    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []

    for pid in train_pids:
        seq_list, labels = data_by_pid[pid]
        X_train_list.extend(seq_list)
        y_train_list.append(labels)

    for pid in test_pids:
        seq_list, labels = data_by_pid[pid]
        X_test_list.extend(seq_list)
        y_test_list.append(labels)

    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # 2) Feature scaling (flatten all time steps for scaler)
    D = X_train_list[0].shape[1]
    X_train_flat = np.vstack([x for x in X_train_list])   # [N_train*T, D]
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    def apply_scaler(seq_list):
        out = []
        for seq in seq_list:
            seq_scaled = scaler.transform(seq)   # [T, D]
            out.append(seq_scaled)
        return out

    X_train_scaled = apply_scaler(X_train_list)
    X_test_scaled  = apply_scaler(X_test_list)

    # 3) Datasets and loaders
    train_ds = SeqDataset(X_train_scaled, y_train)
    test_ds  = SeqDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    # 4) Model, optimizer, loss
    feature_dim = D
    model = TimeSeriesTransformer(
        feature_dim=feature_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None

    # --- NEW: history for plotting ---
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec": [],
        "val_rec": [],
        "val_f1": [],
    }


    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, test_loader, criterion)

        print(f"[{tag}] Epoch {epoch:02d}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, train_f1={train_f1:.3f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, val_prec={val_prec:.3f}, "
              f"val_rec={val_rec:.3f}, val_f1={val_f1:.3f}")

        # --- NEW: log into history ---
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()


    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion)
    print(f"\n=== [{tag}] FINAL TEST PERFORMANCE ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.3f}, Precision: {test_prec:.3f}, "
          f"Recall: {test_rec:.3f}, F1: {test_f1:.3f}\n")

    epochs = history["epoch"]

    # --- Loss vs Epoch ---
    plt.figure(figsize=(4, 3))
    plt.plot(epochs, history["train_loss"], label="Train loss", linestyle="-")
    plt.plot(epochs, history["val_loss"],   label="Val loss",   linestyle="-")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title(f"Loss vs Epoch ({tag})", fontsize=9)

    leg = plt.legend(fontsize=7, loc="upper right")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"learning_curve_loss_{tag}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --- Accuracy vs Epoch ---
    plt.figure(figsize=(4, 3))
    plt.plot(epochs, history["train_acc"], label="Train acc", linestyle="-")
    plt.plot(epochs, history["val_acc"],   label="Val acc",   linestyle="-")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.title(f"Accuracy vs Epoch ({tag})", fontsize=9)

    leg = plt.legend(fontsize=7, loc="lower right")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"learning_curve_acc_{tag}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


    # Confusion matrix – BFRB-style
    y_true, y_pred = get_predictions_and_labels(model, test_loader)
    cm = confusion_matrix(y_true, y_pred)

    unique_labels = np.unique(y_true)
    class_labels = [str(l) for l in unique_labels]   # or ["No CS", "CS"] if you prefer

    out_path = os.path.join(PLOTS_DIR, f"transformer_confusion_matrix_{tag}.png")
    plot_confusion_matrix_compact(
        cm,
        class_labels=class_labels,
        out_path=out_path,
        # title=f"Confusion Matrix ({tag})",
    )



    # Feature importance (only for learned models, not fused)
    if feature_names is not None:
        fi_path = os.path.join(PLOTS_DIR, f"transformer_feature_importance_top10_{tag}.png")
        transformer_feature_importance(model, test_ds, feature_names, topk=10, save_path=fi_path)

    metrics = {
        "accuracy":  test_acc,
        "precision": test_prec,
        "recall":    test_rec,
        "f1":        test_f1,
    }

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "model": model,
        "scaler": scaler,
    }


# =============================================================================
# SSQ CORRELATION ANALYSIS (PER-PARTICIPANT, USING ONLY "Sickness level")
# =============================================================================

def analyze_ssq_correlations(
    data_both,
    model,
    scaler,
    train_pids,
    test_pids,
    ssq_csv_path="CybersicknessDetection_Final - Roller Coaster.csv"
):
    """
    For each participant in data_both:
      - Run combined (IMU+HR) transformer.
      - Compute % / duration of windows predicted sick.
      - Compute % / duration of windows truly sick (labels).
      - Join with SSQ sheet (only 'Sickness level').
      - Save:
          * Pearson & Spearman correlation matrices (predicted vs Sickness level)
          * Scatter plots (Sickness level vs predicted & true metrics)
            with color-coded train/test and different markers for predicted/true.
    """
    # ----- Load SSQ sheet, only needed columns -----
    try:
        ssq_df = pd.read_csv(ssq_csv_path)
    except Exception as e:
        print(f"[SSQ] Could not load SSQ CSV: {e}")
        return

    needed_cols = ["participants", "Sickness level"]
    missing = [c for c in needed_cols if c not in ssq_df.columns]
    if missing:
        print(f"[SSQ] Missing columns in SSQ file: {missing}")
        return

    ssq_df = ssq_df[needed_cols].copy()
    ssq_df["pid"] = ssq_df["participants"].str.strip().str.lower()
    # Fix naming mismatch: "Subham" -> "shubham"
    ssq_df["pid"] = ssq_df["pid"].replace({"subham": "shubham"})

    # ----- Build per-participant summaries for ALL combined-data participants -----
    rows = []
    for pid, (seq_list, labels) in data_both.items():
        # Determine train/test split for this pid
        if pid in train_pids:
            split = "train"
        elif pid in test_pids:
            split = "test"
        else:
            split = "other"

        # Apply the same scaler used during training
        scaled_seqs = [scaler.transform(seq) for seq in seq_list]
        ds_pid = SeqDataset(scaled_seqs, np.array(labels))
        loader_pid = DataLoader(ds_pid, batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn)

        # Get predictions and true labels
        y_true_pid, y_pred_pid = get_predictions_and_labels(model, loader_pid)

        total = len(y_pred_pid)
        if total == 0:
            continue

        # Predicted metrics
        sick_pred = (y_pred_pid == 1).sum()
        cs_percent_pred = 100.0 * sick_pred / total
        cs_duration_pred = sick_pred * WINDOW_SEC

        # True-label metrics
        sick_true = (y_true_pid == 1).sum()
        cs_percent_true = 100.0 * sick_true / total
        cs_duration_true = sick_true * WINDOW_SEC

        rows.append({
            "pid": pid,
            "split": split,  # train / test / other
            "total_windows": total,

            "cs_percent_pred": cs_percent_pred,
            "cs_duration_pred": cs_duration_pred,

            "cs_percent_true": cs_percent_true,
            "cs_duration_true": cs_duration_true,
        })

    if not rows:
        print("[SSQ] No participant-level rows to analyze.")
        return

    model_df = pd.DataFrame(rows)

    # ----- Join with SSQ (only Sickness level) -----
    ssq_cols = ["pid", "Sickness level"]
    merged = pd.merge(model_df, ssq_df[ssq_cols], on="pid", how="inner")

    if merged.empty:
        print("[SSQ] No overlap between participants and SSQ sheet after merge.")
        return

    # ----- Pearson & Spearman correlation (predicted metrics only) -----
    corr_df = merged[["cs_percent_pred", "cs_duration_pred", "Sickness level"]]
    corr_pearson = corr_df.corr(method="pearson")
    corr_spearman = corr_df.corr(method="spearman")

    # Save as CSV
    corr_pearson.to_csv(os.path.join(PLOTS_DIR, "corr_pearson_sickness_level.csv"))
    corr_spearman.to_csv(os.path.join(PLOTS_DIR, "corr_spearman_sickness_level.csv"))

    # Compact heatmaps
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        corr_pearson,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar=False,
        annot_kws={"size": 7},
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    # plt.title("Pearson – Sickness level vs predicted")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "corr_pearson_sickness_level.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        corr_spearman,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar=False,
        annot_kws={"size": 7},
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    # plt.title("Spearman – Sickness level vs predicted")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "corr_spearman_sickness_level.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # =======================================================================
    # SCATTER PLOTS: Sickness level vs model metrics (Predicted & True)
    #   - Color & marker combined into a compact legend
    #   - Legend small and outside the plotting area
    #   - No participant names drawn
    # =======================================================================

    def plot_scatter_compact(x_col, y_pred_col, y_true_col,
                            x_label, y_label, filename):
        plt.figure(figsize=(4, 3))

        # Use only train/test (ignore "other" if any)
        base = merged[merged["split"].isin(["train", "test"])]

        colors = {"train": "C0", "test": "C1"}
        markers = {"pred": "o", "true": "x"}
        label_map = {
            ("train", "pred"): "Train – Predicted",
            ("train", "true"): "Train – True",
            ("test", "pred"):  "Test – Predicted",
            ("test", "true"):  "Test – True",
        }

        for split in ["train", "test"]:
            df_split = base[base["split"] == split]
            if df_split.empty:
                continue

            # Predicted – hollow circles
            plt.scatter(
                df_split[x_col],
                df_split[y_pred_col],
                edgecolors=colors[split],
                facecolors="none",
                marker=markers["pred"],   # 'o'
                s=40,
                alpha=0.9,
                label=label_map[(split, "pred")],
            )

            # True – solid crosses
            plt.scatter(
                df_split[x_col],
                df_split[y_true_col],
                c=colors[split],
                marker=markers["true"],   # 'x'
                s=40,
                alpha=0.9,
                label=label_map[(split, "true")],
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Legend INSIDE plot, bottom-right, small & bold font
        leg = plt.legend(
            fontsize=7,          # small font
            loc="lower right",   # inside, bottom-right corner
            borderaxespad=0.5,
            framealpha=0.8,
        )

        # Make legend text bold
        for text in leg.get_texts():
            text.set_fontweight("bold")

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


    # 1) Sickness level vs % time cybersick
    plot_scatter_compact(
        x_col="Sickness level",
        y_pred_col="cs_percent_pred",
        y_true_col="cs_percent_true",
        x_label="SSQ Sickness level",
        y_label="% time cybersickness",
        filename="scatter_sickness_level_vs_percent_pred_true.png",
    )

    # 2) Sickness level vs total cybersickness duration (seconds)
    plot_scatter_compact(
        x_col="Sickness level",
        y_pred_col="cs_duration_pred",
        y_true_col="cs_duration_true",
        x_label="SSQ Sickness level",
        y_label="Total cybersickness time (s)",
        filename="scatter_sickness_level_vs_duration_pred_true.png",
    )

    print("[SSQ] Saved compact scatter plots and correlation matrices (Sickness level only) in", PLOTS_DIR)


# =============================================================================
# MAIN: TRAIN 3 MODELS + FUSED (AND) ENSEMBLE
# =============================================================================

def main():
    # 1) Build three datasets: IMU-only, HR-only, Combined
    data_imu, feat_imu   = build_all_sequences(mode="imu",  method=HR_INTERP_METHOD)
    data_hr,  feat_hr    = build_all_sequences(mode="hr",   method=HR_INTERP_METHOD)
    data_both, feat_both = build_all_sequences(mode="both", method=HR_INTERP_METHOD)

    # ---------- BOXPLOTS: FEATURE vs CYBERSICKNESS ----------
    # Combine all IMU windows across participants into one DataFrame
    all_imu_X = []
    all_imu_y = []
    for pid, (seq_list, labels) in data_imu.items():
        # seq_list elements are [T, D]; with WINDOW_SEC=1, T=1, so squeeze
        X_pid = np.stack(seq_list, axis=0).squeeze(1)  # [N, 1, D] -> [N, D]
        all_imu_X.append(X_pid)
        all_imu_y.append(labels)

    if all_imu_X:
        all_imu_X = np.vstack(all_imu_X)        # [N_total, D]
        all_imu_y = np.concatenate(all_imu_y)   # [N_total]

        df_imu = pd.DataFrame(all_imu_X, columns=feat_imu)
        df_imu["Cybersickness"] = all_imu_y

        # IMU boxplot: sway_var vs Cybersickness (instead of movement_energy)
        if "sway_var" in df_imu.columns:
            plt.figure(figsize=(3, 3))
            sns.boxplot(
                data=df_imu,
                x="Cybersickness",
                y="sway_var",
            )
            plt.xlabel("Cybersickness (0 = No, 1 = Yes)", fontsize=8)
            plt.ylabel("Sway variance", fontsize=8)
            # plt.title("IMU: Sway variance vs Cybersickness", fontsize=9)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, "boxplot_imu_sway_var_vs_cybersickness.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    # HR feature boxplot (e.g., hr vs Cybersickness)
    all_hr_X = []
    all_hr_y = []
    for pid, (seq_list, labels) in data_hr.items():
        X_pid = np.stack(seq_list, axis=0).squeeze(1)  # [N, 1, D] -> [N, D]
        all_hr_X.append(X_pid)
        all_hr_y.append(labels)

    if all_hr_X:
        all_hr_X = np.vstack(all_hr_X)
        all_hr_y = np.concatenate(all_hr_y)

        df_hr = pd.DataFrame(all_hr_X, columns=feat_hr)
        df_hr["Cybersickness"] = all_hr_y

        if "hr" in df_hr.columns:
            plt.figure(figsize=(3, 3))
            sns.boxplot(
                data=df_hr,
                x="Cybersickness",
                y="hr",
            )
            plt.xlabel("Cybersickness (0 = No, 1 = Yes)", fontsize=8)
            plt.ylabel("Heart rate (bpm)", fontsize=8)
            # plt.title("HR: BPM vs Cybersickness", fontsize=9)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOTS_DIR, "boxplot_hr_bpm_vs_cybersickness.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


    # Participants that exist in all three datasets
    common_pids = sorted(set(data_imu.keys()) & set(data_hr.keys()) & set(data_both.keys()))
    print("\nParticipants with usable sequences in ALL modes:", common_pids)

    if len(common_pids) < 3:
        print("Not enough common participants for a  (train, test) split. Exiting.")
        return

    # Same train/test split across the three models
    train_pids = common_pids[:max(1, len(common_pids) - 3)]
    test_pids  = common_pids[max(1, len(common_pids) - 3):]

    print("Train participants:", train_pids)
    print("Test participants: ", test_pids)

    # 2) Train / evaluate three transformer models
    res_imu   = train_transformer_for_dataset(data_imu,   feat_imu,   train_pids, test_pids, tag="imu_only")
    res_hr    = train_transformer_for_dataset(data_hr,    feat_hr,    train_pids, test_pids, tag="hr_only")
    res_both  = train_transformer_for_dataset(data_both,  feat_both,  train_pids, test_pids, tag="combined")

    # 2b) SSQ correlation analysis (ALL participants, Sickness level only)
    analyze_ssq_correlations(
        data_both=data_both,
        model=res_both["model"],
        scaler=res_both["scaler"],
        train_pids=train_pids,
        test_pids=test_pids,
        ssq_csv_path="CybersicknessDetection_Final - Roller Coaster.csv",
    )

    # 2c) LOOCV for COMBINED model only
    print("\n================= RUNNING LOOCV ON COMBINED MODEL =================\n")
    run_loocv_combined(data_both, feat_both)

    # 2d) Longevity experiment (same combined model, new day-wise recordings)
    print("\n================= RUNNING LONGEVITY EXPERIMENT =================\n")
    run_longevity_experiment(
        model=res_both["model"],
        scaler=res_both["scaler"],
        expected_feature_names=feat_both,
        window_sec=WINDOW_SEC,
        sampling_hz=SAMPLING_HZ,
        longevity_dir="."   # folder where k1/a1/k2 files live
    )



    # 3) Fused (AND) decision:
    #    predict 1 only if BOTH (IMU-only and HR-only) predict 1, else 0.
    y_true_imu, y_pred_imu = res_imu["y_true"], res_imu["y_pred"]
    y_true_hr,  y_pred_hr  = res_hr["y_true"],  res_hr["y_pred"]

    # Align lengths by min (since sequence counts should be very close)
    L = min(len(y_true_imu), len(y_true_hr))
    y_true_fused = y_true_imu[:L]   # they should be same labels
    fused_pred = ((y_pred_imu[:L] == 1) & (y_pred_hr[:L] == 1)).astype(int)

    fused_acc  = accuracy_score(y_true_fused, fused_pred)
    fused_prec = precision_score(y_true_fused, fused_pred, zero_division=0)
    fused_rec  = recall_score(y_true_fused, fused_pred, zero_division=0)
    fused_f1   = f1_score(y_true_fused, fused_pred, zero_division=0)

    print("=== [FUSED AND] PERFORMANCE (IMU-only ∧ HR-only) ===")
    print(f"Accuracy: {fused_acc:.3f}, Precision: {fused_prec:.3f}, "
          f"Recall: {fused_rec:.3f}, F1: {fused_f1:.3f}")

    # Confusion matrix for fused model (compact + bold)
    cm_fused = confusion_matrix(y_true_fused, fused_pred)
    labels = np.unique(y_true_fused)

    plt.figure(figsize=(3.5, 3.0))
    ax = sns.heatmap(
        cm_fused,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12, "weight": "bold"},
    )

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    # ⛔ remove title
    # ax.set_title("Confusion Matrix (Fused AND)", fontsize=9)

    ax.tick_params(axis="both", labelsize=11)



    # 4) Comparison bar chart across the four models

    comp = { "IMU-only": res_imu["metrics"], 
            "HR-only": res_hr["metrics"], 
            "Combined": res_both["metrics"], 
            # "Fused-AND": { 
            # # "accuracy": fused_acc, 
            # # "precision": fused_prec, 
            # # "recall": fused_rec, 
            # # "f1": fused_f1, 
            # # }, 
            }
    # -----------------------------------------------------------
    # Styled Model Comparison Plot (IMU-only, HR-only, Combined)
    # -----------------------------------------------------------

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
        "figure.dpi": 180,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })

    # Consistent colors + hatches (extended to 4 metrics)
    COLORS  = ["#4e79a7", "#e15759", "#7c48ff", "#59a14f"]   # 4 distinct colors
    HATCHES = ["////", "\\\\\\\\", "....", "xx"]             # 4 distinct hatches

    MODELS = ["IMU-only", "HR-only", "Combined"]
    METRICS = ["accuracy", "precision", "recall", "f1"]
    METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-score"]

    # DataFrame extracted from your results
    metrics_df = pd.DataFrame(comp).T[["accuracy", "precision", "recall", "f1"]]
    core_metrics_df = metrics_df.loc[MODELS]

    x = np.arange(len(MODELS))
    width = 0.18   # thinner bars, closer spacing

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        offset = (i - (len(METRICS) - 1) / 2) * width
        values = core_metrics_df[metric].values

        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[i],
            edgecolor="black",
            linewidth=1.0,
            hatch=HATCHES[i],
        )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=10, ha="center")

    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",              # position relative to bbox_to_anchor
        bbox_to_anchor=(0.5, 0.83),      # (x, y) in axes coords → middle, a bit above HR-only
        ncols=2,
        frameon=True,
        facecolor="white",
        borderaxespad=0.4,
    )


    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "transformer_models_comparison_styled.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


    # --- Separate bar plots for each metric: IMU vs HR vs Combined ---
    core_models = ["IMU-only", "HR-only", "Combined"]
    core_metrics_df = metrics_df.loc[core_models]

    # x positions tightly packed
    x = np.arange(len(core_models)) * 0.4     # default would be 1.0 apart → now closer

    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.figure(figsize=(3, 3))

        values = core_metrics_df[metric].values

        # thinner bars + reduced spacing
        plt.bar(x, values, width=0.1)

        plt.ylim(0.0, 1.0)
        plt.ylabel(metric.capitalize())
        # plt.title(f"{metric.capitalize()} – IMU vs HR vs Combined")

        # Apply model names to the reduced x-positions
        plt.xticks(x, core_models, rotation=15)

        plt.tight_layout()
        out_name = f"transformer_{metric}_imu_hr_combined.png"
        plt.savefig(os.path.join(PLOTS_DIR, out_name), dpi=300, bbox_inches="tight")
        plt.close()



    print(f"\nAll confusion matrices and comparison plot saved in '{PLOTS_DIR}'.")


if __name__ == "__main__":
    main()
