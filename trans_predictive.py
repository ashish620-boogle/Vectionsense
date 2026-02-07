import os
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# Predictive window settings
# Input: [t - LOOKBACK_SEC, t]
# Output: cybersickness at t + HORIZON_SEC (optional FUTURE_WINDOW_SEC aggregation)
LOOKBACK_SEC = 4
HORIZON_SEC = 0
FUTURE_WINDOW_SEC = 1
SAMPLING_HZ = 1
STRIDE_SEC = 1  # sliding window stride; set to LOOKBACK_SEC for non-overlap

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
LATENT_LOSS_WEIGHT = 1.0
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 1e-4
LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 3
LR_SCHED_MIN_LR = 1e-6
AUX_TARGET = "latent"  # "raw_seq", "latent", "raw_mean", "both", "none"
AUX_METRICS_EVERY = 1
LATENT_PLOT_EVERY = 10
LATENT_PLOT_MAX_POINTS = 300

MODELS_DIR = "models_transformer"
os.makedirs(MODELS_DIR, exist_ok=True)
PLOTS_DIR = "plots_predictive"
os.makedirs(PLOTS_DIR, exist_ok=True)
EVAL_PLOTS_DIR = "plots_predictive_eval"
os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
PREDICTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "PREDICTIVE_MODEL.pth")
PROPOSED_MODEL_PATH = os.path.join(MODELS_DIR, "PROPOSED_MODEL.pth")
USE_PROPOSED_MODEL = True
USE_PROPOSED_SCALER = True
ACC_EVERY = 1

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


def apply_scaler(seq_list, scaler):
    """
    Apply a fitted scaler to a list of [T, D] sequences.
    """
    scaled = []
    for seq in seq_list:
        T, D = seq.shape
        flat = seq.reshape(-1, D)
        flat_scaled = scaler.transform(flat)
        scaled.append(flat_scaled.reshape(T, D))
    return scaled


def load_proposed_model(feature_dim, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] PROPOSED_MODEL not found at {checkpoint_path}.")
        return None, None, None

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TimeSeriesTransformer(
        feature_dim=feature_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    scaler = ckpt.get("scaler", None)
    feature_names = ckpt.get("feature_names", None)
    return model, scaler, feature_names

# =============================================================================
# BUILD PREDICTIVE SEQUENCES
# =============================================================================

def build_predictive_sequences_for_pid(
    pid,
    imu_path,
    hr_path_aligned=None,
    lookback_sec=1,
    horizon_sec=1,
    future_window_sec=1,
    sampling_hz=1,
    stride_sec=1,
    mode="both",
):
    """
    Build predictive sequences for a single participant.

    Input window:  [t - lookback, t]
    Output label:  cybersickness at t + horizon (aggregated over future_window)

    mode:
        "imu"  -> only IMU features
        "hr"   -> only HR features (requires hr_path_aligned)
        "both" -> concatenated IMU + HR features

    Returns:
        past_list:   list of [T_past, D]
        future_list: list of [T_future, D]
        labels:      np.array of int
        feature_names: list of D column names
    """
    if not os.path.exists(imu_path):
        print(f"[WARN] IMU file missing for {pid}: {imu_path}")
        return [], [], np.array([]), None

    imu_df = pd.read_csv(imu_path)
    imu_df = ensure_cybersickness_column(imu_df)
    imu_df = drop_non_feature_columns(imu_df, "Cybersickness")
    imu_df = extract_imu_features(imu_df)
    imu_df = drop_non_feature_columns(imu_df, "Cybersickness")

    label_series = None
    X_all = None
    feature_names = None

    if hr_path_aligned is not None and os.path.exists(hr_path_aligned):
        hr_df = pd.read_csv(hr_path_aligned)
        hr_df = ensure_cybersickness_column(hr_df)
        hr_df = drop_non_feature_columns(hr_df, "Cybersickness")
        hr_df = extract_hr_features(hr_df)
        hr_df = drop_non_feature_columns(hr_df, "Cybersickness")

        # Align lengths
        L = min(len(imu_df), len(hr_df))
        imu_df = imu_df.iloc[:L].reset_index(drop=True)
        hr_df = hr_df.iloc[:L].reset_index(drop=True)

        label_series = imu_df["Cybersickness"].values

        X_imu_df = imu_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])
        X_hr_df = hr_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])

        X_imu = X_imu_df.values
        X_hr = X_hr_df.values

        imu_cols = list(X_imu_df.columns)
        hr_cols = list(X_hr_df.columns)

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
        if mode == "hr":
            print(f"[WARN] No HR aligned file for {pid}, skipping in HR-only mode.")
            return [], [], np.array([]), None

        label_series = imu_df["Cybersickness"].values
        X_imu_df = imu_df.drop(columns=["Cybersickness"]).select_dtypes(include=[np.number])
        X_all = X_imu_df.values
        feature_names = list(X_imu_df.columns)

    # Windowing
    lookback_size = int(lookback_sec * sampling_hz)
    horizon_size = int(horizon_sec * sampling_hz)
    future_size = int(future_window_sec * sampling_hz)
    stride = int(stride_sec * sampling_hz)

    if lookback_size <= 0 or future_size <= 0 or stride <= 0:
        raise ValueError("lookback_sec, future_window_sec, and stride_sec must be > 0")

    total_needed = lookback_size + horizon_size + future_size
    rows = len(label_series)

    past_list = []
    future_list = []
    label_list = []

    for start in range(0, rows - total_needed + 1, stride):
        past_start = start
        past_end = start + lookback_size
        future_start = past_end + horizon_size
        future_end = future_start + future_size

        X_past = X_all[past_start:past_end]
        X_future = X_all[future_start:future_end]
        y_future = label_series[future_start:future_end]

        if X_past.shape[0] != lookback_size or X_future.shape[0] != future_size:
            continue

        label = int(y_future.mean() >= 0.5)
        past_list.append(X_past.astype(np.float32))
        future_list.append(X_future.astype(np.float32))
        label_list.append(label)

    return past_list, future_list, np.array(label_list, dtype=np.int64), feature_names


def build_all_predictive_sequences(mode="both", method="cubic"):
    """
    Build predictive sequences for all participants for a given mode:
      - "imu"
      - "hr"
      - "both"

    Returns:
        data_by_pid: {pid: (past_list, future_list, labels)}
        feature_names: list of feature names
    """
    data_by_pid = {}
    global_feature_names = None

    hr_map = get_hr_aligned_files(method)

    for pid, imu_path in imu_files.items():
        hr_path = hr_map.get(pid, None)

        past_list, future_list, labels, feat_names = build_predictive_sequences_for_pid(
            pid,
            imu_path,
            hr_path_aligned=hr_path,
            lookback_sec=LOOKBACK_SEC,
            horizon_sec=HORIZON_SEC,
            future_window_sec=FUTURE_WINDOW_SEC,
            sampling_hz=SAMPLING_HZ,
            stride_sec=STRIDE_SEC,
            mode=mode,
        )

        if len(past_list) == 0:
            continue

        data_by_pid[pid] = (past_list, future_list, labels)
        print(f"[INFO] {mode.upper()} - {pid}: {len(past_list)} predictive sequences")

        if global_feature_names is None and feat_names is not None:
            global_feature_names = feat_names
        elif feat_names is not None and feat_names != global_feature_names:
            print(f"[WARN] Feature name mismatch for {pid} in mode={mode}; "
                  f"using feature_names from first participant.")

    return data_by_pid, global_feature_names

# =============================================================================
# DATASET
# =============================================================================

class PredictiveSeqDataset(Dataset):
    def __init__(self, past_list, future_list, label_array):
        self.past_list = past_list
        self.future_list = future_list
        self.labels = label_array

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_past = self.past_list[idx]
        x_future = self.future_list[idx]
        y = self.labels[idx]
        return x_past, x_future, y


def collate_fn(batch):
    pasts, futures, ys = zip(*batch)
    pasts = torch.tensor(np.stack(pasts, axis=0), dtype=torch.float32)    # [B, T_past, D]
    futures = torch.tensor(np.stack(futures, axis=0), dtype=torch.float32)  # [B, T_future, D]
    ys = torch.tensor(ys, dtype=torch.long)
    return pasts, futures, ys

# =============================================================================
# TRANSFORMER BACKBONE
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
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerEncoderBackbone(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
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

    def forward(self, x):
        x = self.input_proj(x)      # [B, T, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)         # [B, T, d_model]
        x = x.mean(dim=1)           # [B, d_model]
        return x


class TimeSeriesTransformer(nn.Module):
    """
    Same architecture as PROPOSED_MODEL in trans.py.
    Provides encode() and classify_latent() helpers.
    """
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

    def encode(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return x

    def classify_latent(self, latent):
        return self.cls_head(latent)

    def forward(self, x):
        latent = self.encode(x)
        logits = self.cls_head(latent)
        return logits


class PredictiveTransformer(nn.Module):
    """
    Past window -> predict future sensor sequence.
    """
    def __init__(self, feature_dim, future_steps, d_model=64, nhead=4, num_layers=2,
                 num_classes=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.future_steps = future_steps
        self.encoder = TransformerEncoderBackbone(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.future_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.future_seq_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, future_steps * feature_dim),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_past, x_future=None, return_targets=False, teacher_encoder=None):
        past_repr = self.encoder(x_past)
        pred_future_repr = self.future_predictor(past_repr)
        pred_future_seq = self.future_seq_head(pred_future_repr)
        pred_future_seq = pred_future_seq.view(
            x_past.size(0), self.future_steps, self.feature_dim
        )

        if x_future is None or not return_targets:
            return pred_future_repr, pred_future_seq, None, None

        with torch.no_grad():
            if teacher_encoder is not None:
                true_future_repr = teacher_encoder(x_future)
            else:
                true_future_repr = self.encoder(x_future)
            true_future_seq = x_future

        return pred_future_repr, pred_future_seq, true_future_repr, true_future_seq

# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_epoch(model, loader, optimizer, mse_loss, latent_weight, aux_target, teacher_encoder=None):
    model.train()
    total_loss = 0.0

    for X_past, X_future, _ in loader:
        X_past = X_past.to(device)
        X_future = X_future.to(device)

        optimizer.zero_grad()
        pred_lat, pred_seq, true_lat, true_seq = model(
            X_past, X_future, return_targets=True, teacher_encoder=teacher_encoder
        )
        loss_aux = 0.0
        if aux_target == "raw_seq":
            loss_aux = mse_loss(pred_seq, true_seq)
        elif aux_target == "raw_mean":
            loss_aux = mse_loss(pred_seq.mean(dim=1), true_seq.mean(dim=1))
        elif aux_target == "latent":
            loss_aux = mse_loss(pred_lat, true_lat)
        elif aux_target == "both":
            loss_aux = mse_loss(pred_seq, true_seq) + mse_loss(pred_lat, true_lat)
        elif aux_target == "none":
            loss_aux = 0.0
        else:
            raise ValueError(f"Unknown AUX_TARGET: {aux_target}")

        loss = latent_weight * loss_aux

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_past.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def evaluate_pred_loss(model, loader, mse_loss, latent_weight, aux_target, teacher_encoder=None):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_past, X_future, _ in loader:
            X_past = X_past.to(device)
            X_future = X_future.to(device)

            pred_lat, pred_seq, true_lat, true_seq = model(
                X_past, X_future, return_targets=True, teacher_encoder=teacher_encoder
            )

            if aux_target == "raw_seq":
                loss_aux = mse_loss(pred_seq, true_seq)
            elif aux_target == "raw_mean":
                loss_aux = mse_loss(pred_seq.mean(dim=1), true_seq.mean(dim=1))
            elif aux_target == "latent":
                loss_aux = mse_loss(pred_lat, true_lat)
            elif aux_target == "both":
                loss_aux = mse_loss(pred_seq, true_seq) + mse_loss(pred_lat, true_lat)
            elif aux_target == "none":
                loss_aux = 0.0
            else:
                raise ValueError(f"Unknown AUX_TARGET: {aux_target}")

            loss = latent_weight * loss_aux
            total_loss += loss.item() * X_past.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def get_predictions_with_proposed(model, proposed_model, loader):
    model.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for X_past, _, y in loader:
            X_past = X_past.to(device)
            y = y.to(device)
            pred_lat, _, _, _ = model(X_past, return_targets=False)
            if proposed_model is not None:
                logits = proposed_model.classify_latent(pred_lat)
            else:
                logits = model.cls_head(pred_lat)
            preds = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_pred)


def compute_aux_metrics(model, loader, teacher_encoder=None):
    model.eval()
    lat_mse_sum = 0.0
    seq_mse_sum = 0.0
    lat_cos_sum = 0.0
    total = 0

    with torch.no_grad():
        for X_past, X_future, _ in loader:
            X_past = X_past.to(device)
            X_future = X_future.to(device)

            pred_lat, pred_seq, true_lat, true_seq = model(
                X_past, X_future, return_targets=True, teacher_encoder=teacher_encoder
            )
            bs = X_past.size(0)

            lat_mse_sum += F.mse_loss(pred_lat, true_lat, reduction="mean").item() * bs
            seq_mse_sum += F.mse_loss(pred_seq, true_seq, reduction="mean").item() * bs
            lat_cos_sum += F.cosine_similarity(pred_lat, true_lat, dim=1).mean().item() * bs
            total += bs

    if total == 0:
        return None, None, None

    return lat_mse_sum / total, seq_mse_sum / total, lat_cos_sum / total


def plot_aux_history(history, out_path):
    if not history["epoch"]:
        return

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax[0].plot(history["epoch"], history["latent_mse"], label="latent_mse")
    ax[0].plot(history["epoch"], history["seq_mse"], label="seq_mse")
    ax[0].set_ylabel("MSE")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(history["epoch"], history["latent_cos"], label="latent_cos")
    ax[1].set_ylabel("Cosine sim")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latent_alignment(model, loader, out_path, max_points=300, teacher_encoder=None):
    model.eval()
    pred_list = []
    true_list = []

    with torch.no_grad():
        for X_past, X_future, _ in loader:
            X_past = X_past.to(device)
            X_future = X_future.to(device)
            pred_lat, _, true_lat, _ = model(
                X_past, X_future, return_targets=True, teacher_encoder=teacher_encoder
            )
            pred_list.append(pred_lat.cpu().numpy())
            true_list.append(true_lat.cpu().numpy())
            if sum(p.shape[0] for p in pred_list) >= max_points:
                break

    if not pred_list:
        return

    pred = np.vstack(pred_list)
    true = np.vstack(true_list)
    if pred.shape[0] < 2:
        return
    if pred.shape[0] > max_points:
        idx = np.random.choice(pred.shape[0], size=max_points, replace=False)
        pred = pred[idx]
        true = true[idx]

    Z = PCA(n_components=2).fit_transform(np.vstack([pred, true]))
    pred_z = Z[:pred.shape[0]]
    true_z = Z[pred.shape[0]:]

    plt.figure(figsize=(5, 5))
    plt.scatter(true_z[:, 0], true_z[:, 1], s=18, alpha=0.7, label="true_future")
    plt.scatter(pred_z[:, 0], pred_z[:, 1], s=18, alpha=0.7, label="pred_future")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss_curves(history, out_path):
    if not history["epoch"]:
        return

    plt.figure(figsize=(4, 3))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss", linestyle="-")
    plt.plot(history["epoch"], history["test_loss"],  label="Test loss",  linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    leg = plt.legend(fontsize=7, loc="upper right")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curves(history, out_path):
    if not history["epoch"]:
        return

    plt.figure(figsize=(4, 3))
    plt.plot(history["epoch"], history["train_acc"], label="Train acc", linestyle="-")
    plt.plot(history["epoch"], history["test_acc"],  label="Test acc",  linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    leg = plt.legend(fontsize=7, loc="lower right")
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(metrics_dict, out_path):
    labels = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in labels]

    plt.figure(figsize=(4.5, 3.2))
    colors = ["#4e79a7", "#e15759", "#7c48ff", "#59a14f"]  # Accuracy, Precision, Recall, F1
    hatches = ["////", "\\\\\\\\", "....", "xx"]
    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_labels, out_path, title=None):
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

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Build predictive sequences (combined IMU + HR)
    data_both, feat_both = build_all_predictive_sequences(mode="both", method=HR_INTERP_METHOD)

    common_pids = sorted(data_both.keys())
    print("\nParticipants with usable predictive sequences:", common_pids)

    if len(common_pids) < 3:
        print("Not enough participants for a train/test split. Exiting.")
        return

    # Same train/test split style as trans.py
    train_pids = common_pids[:max(1, len(common_pids) - 3)]
    test_pids = common_pids[max(1, len(common_pids) - 3):]

    print("Train participants:", train_pids)
    print("Test participants: ", test_pids)

    # Gather sequences
    X_train_past, X_train_future, y_train_list = [], [], []
    X_test_past, X_test_future, y_test_list = [], [], []

    for pid in train_pids:
        past_list, future_list, labels = data_both[pid]
        X_train_past.extend(past_list)
        X_train_future.extend(future_list)
        y_train_list.append(labels)

    for pid in test_pids:
        past_list, future_list, labels = data_both[pid]
        X_test_past.extend(past_list)
        X_test_future.extend(future_list)
        y_test_list.append(labels)

    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Load PROPOSED_MODEL (used as teacher encoder + classifier head)
    feature_dim = X_train_past[0].shape[1]
    proposed_model, proposed_scaler, proposed_feat_names = (None, None, None)
    if USE_PROPOSED_MODEL:
        proposed_model, proposed_scaler, proposed_feat_names = load_proposed_model(
            feature_dim, PROPOSED_MODEL_PATH
        )

        if proposed_feat_names is not None and proposed_feat_names != feat_both:
            print("[WARN] Feature name mismatch vs PROPOSED_MODEL. "
                  "Ensure feature extraction order matches.")
        if proposed_model is not None:
            print("[INFO] Using PROPOSED_MODEL encoder + classifier head.")
        else:
            print("[WARN] PROPOSED_MODEL unavailable. Falling back to predictive head.")

    # Fit or reuse scaler
    if USE_PROPOSED_SCALER and proposed_scaler is not None:
        scaler = proposed_scaler
        print("[INFO] Using PROPOSED_MODEL scaler for predictive model.")
    else:
        scaler = StandardScaler()
        X_flat = np.vstack([s.reshape(-1, s.shape[-1]) for s in (X_train_past + X_train_future)])
        scaler.fit(X_flat)

    # Apply scaler
    X_train_past = apply_scaler(X_train_past, scaler)
    X_train_future = apply_scaler(X_train_future, scaler)
    X_test_past = apply_scaler(X_test_past, scaler)
    X_test_future = apply_scaler(X_test_future, scaler)

    # Datasets and loaders
    train_ds = PredictiveSeqDataset(X_train_past, X_train_future, y_train)
    test_ds = PredictiveSeqDataset(X_test_past, X_test_future, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    future_steps = X_train_future[0].shape[0]
    model = PredictiveTransformer(
        feature_dim=feature_dim,
        future_steps=future_steps,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_classes=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHED_FACTOR,
        patience=LR_SCHED_PATIENCE,
        min_lr=LR_SCHED_MIN_LR,
    )
    mse_loss = nn.MSELoss()
    teacher_encoder = proposed_model.encode if proposed_model is not None else None

    print("\n================= TRAINING PREDICTIVE TRANSFORMER =================\n")
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    aux_history = {"epoch": [], "latent_mse": [], "seq_mse": [], "latent_cos": []}
    train_history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "train_f1": [],
        "test_f1": [],
    }

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            mse_loss,
            LATENT_LOSS_WEIGHT,
            AUX_TARGET,
            teacher_encoder=teacher_encoder,
        )
        te_loss = evaluate_pred_loss(
            model,
            test_loader,
            mse_loss,
            LATENT_LOSS_WEIGHT,
            AUX_TARGET,
            teacher_encoder=teacher_encoder,
        )

        if ACC_EVERY > 0 and (epoch % ACC_EVERY == 0):
            y_true_tr, y_pred_tr = get_predictions_with_proposed(model, proposed_model, train_loader)
            tr_acc = accuracy_score(y_true_tr, y_pred_tr)
            tr_f1 = f1_score(y_true_tr, y_pred_tr, zero_division=0)
        else:
            tr_acc = float("nan")
            tr_f1 = float("nan")

        y_true, y_pred = get_predictions_with_proposed(model, proposed_model, test_loader)
        te_acc = accuracy_score(y_true, y_pred)
        te_prec = precision_score(y_true, y_pred, zero_division=0)
        te_rec = recall_score(y_true, y_pred, zero_division=0)
        te_f1 = f1_score(y_true, y_pred, zero_division=0)

        scheduler.step(te_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_repr_loss={tr_loss:.4f} | "
            f"test_repr_loss={te_loss:.4f}, test_acc={te_acc:.3f}, test_f1={te_f1:.3f} | "
            f"lr={current_lr:.2e}"
        )

        train_history["epoch"].append(epoch)
        train_history["train_loss"].append(tr_loss)
        train_history["test_loss"].append(te_loss)
        train_history["train_acc"].append(tr_acc)
        train_history["test_acc"].append(te_acc)
        train_history["train_f1"].append(tr_f1)
        train_history["test_f1"].append(te_f1)

        if AUX_METRICS_EVERY > 0 and (epoch % AUX_METRICS_EVERY == 0):
            lat_mse, seq_mse, lat_cos = compute_aux_metrics(
                model, test_loader, teacher_encoder=teacher_encoder
            )
            if lat_mse is not None:
                aux_history["epoch"].append(epoch)
                aux_history["latent_mse"].append(lat_mse)
                aux_history["seq_mse"].append(seq_mse)
                aux_history["latent_cos"].append(lat_cos)
                print(
                    f"[AUX] latent_mse={lat_mse:.4f} | seq_mse={seq_mse:.4f} | "
                    f"latent_cos={lat_cos:.4f}"
                )

        if LATENT_PLOT_EVERY > 0 and (epoch % LATENT_PLOT_EVERY == 0):
            out_path = os.path.join(PLOTS_DIR, f"latent_alignment_epoch_{epoch:03d}.png")
            plot_latent_alignment(
                model,
                test_loader,
                out_path,
                max_points=LATENT_PLOT_MAX_POINTS,
                teacher_encoder=teacher_encoder,
            )

        if te_loss < (best_loss - EARLY_STOP_MIN_DELTA):
            best_loss = te_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"[EARLY STOP] No improvement in test representation loss for "
                    f"{EARLY_STOP_PATIENCE} epochs. Best epoch: {best_epoch} "
                    f"(loss={best_loss:.4f})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    plot_aux_history(aux_history, os.path.join(PLOTS_DIR, "aux_metrics_over_epochs.png"))
    plot_loss_curves(train_history, os.path.join(EVAL_PLOTS_DIR, "loss_curve.png"))
    plot_accuracy_curves(train_history, os.path.join(EVAL_PLOTS_DIR, "accuracy_curve.png"))

    # Final evaluation
    y_true, y_pred = get_predictions_with_proposed(model, proposed_model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== [PREDICTIVE MODEL PERFORMANCE] ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")

    # Performance metrics bar chart
    plot_metrics_bar(
        {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        os.path.join(EVAL_PLOTS_DIR, "performance_metrics.png"),
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        class_labels=[0, 1],
        out_path=os.path.join(EVAL_PLOTS_DIR, "confusion_matrix.png"),
    )

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "feature_names": feat_both,
        "train_pids": train_pids,
        "test_pids": test_pids,
        "lookback_sec": LOOKBACK_SEC,
        "horizon_sec": HORIZON_SEC,
        "future_window_sec": FUTURE_WINDOW_SEC,
        "sampling_hz": SAMPLING_HZ,
        "stride_sec": STRIDE_SEC,
        "latent_loss_weight": LATENT_LOSS_WEIGHT,
        "aux_target": AUX_TARGET,
        "use_proposed_model": USE_PROPOSED_MODEL,
        "use_proposed_scaler": USE_PROPOSED_SCALER,
    }
    torch.save(checkpoint, PREDICTIVE_MODEL_PATH)
    print(f"[PREDICTIVE] Saved model to {PREDICTIVE_MODEL_PATH}")


if __name__ == "__main__":
    main()
