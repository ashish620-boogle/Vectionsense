import os
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
LOOKBACK_SEC = 10
HORIZON_SEC = 0
FUTURE_WINDOW_SEC = 1
SAMPLING_HZ = 1
STRIDE_SEC = 1  # sliding window stride; set to LOOKBACK_SEC for non-overlap

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
LATENT_LOSS_WEIGHT = 0.03
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 1e-4
LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 3
LR_SCHED_MIN_LR = 1e-6

MODELS_DIR = "models_transformer"
os.makedirs(MODELS_DIR, exist_ok=True)
PREDICTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "PREDICTIVE_MODEL.pth")

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


class PredictiveTransformer(nn.Module):
    """
    Past window -> predict future latent -> classify future state.
    """
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2,
                 num_classes=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
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
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_past, x_future=None, return_latent=False):
        past_repr = self.encoder(x_past)
        pred_future_repr = self.future_predictor(past_repr)
        logits = self.cls_head(pred_future_repr)

        if x_future is None or not return_latent:
            return logits, pred_future_repr, None

        with torch.no_grad():
            true_future_repr = self.encoder(x_future)

        return logits, pred_future_repr, true_future_repr

# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_epoch(model, loader, optimizer, ce_loss, mse_loss, latent_weight):
    model.train()
    total_loss = 0.0
    all_y = []
    all_pred = []

    for X_past, X_future, y in loader:
        X_past = X_past.to(device)
        X_future = X_future.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, pred_repr, true_repr = model(X_past, X_future, return_latent=True)

        loss_cls = ce_loss(logits, y)
        loss_lat = mse_loss(pred_repr, true_repr)
        loss = loss_cls + latent_weight * loss_lat

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_past.size(0)
        preds = logits.argmax(dim=1)
        all_y.append(y.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, zero_division=0)
    return avg_loss, acc, f1


def evaluate(model, loader, ce_loss):
    model.eval()
    total_loss = 0.0
    all_y = []
    all_pred = []

    with torch.no_grad():
        for X_past, _, y in loader:
            X_past = X_past.to(device)
            y = y.to(device)

            logits, _, _ = model(X_past, return_latent=False)
            loss = ce_loss(logits, y)
            total_loss += loss.item() * X_past.size(0)

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


def get_predictions(model, loader):
    model.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for X_past, _, y in loader:
            X_past = X_past.to(device)
            y = y.to(device)
            logits, _, _ = model(X_past, return_latent=False)
            preds = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_pred)

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

    # Fit scaler on training past + future windows (same feature space)
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
    feature_dim = X_train_past[0].shape[1]
    model = PredictiveTransformer(
        feature_dim=feature_dim,
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
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    print("\n================= TRAINING PREDICTIVE TRANSFORMER =================\n")
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, LATENT_LOSS_WEIGHT
        )
        te_loss, te_acc, te_prec, te_rec, te_f1 = evaluate(model, test_loader, ce_loss)

        scheduler.step(te_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.3f}, train_f1={tr_f1:.3f} | "
            f"test_loss={te_loss:.4f}, test_acc={te_acc:.3f}, test_f1={te_f1:.3f} | "
            f"lr={current_lr:.2e}"
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
                    f"[EARLY STOP] No improvement in test loss for "
                    f"{EARLY_STOP_PATIENCE} epochs. Best epoch: {best_epoch} "
                    f"(loss={best_loss:.4f})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    y_true, y_pred = get_predictions(model, test_loader)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== [PREDICTIVE MODEL PERFORMANCE] ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")

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
    }
    torch.save(checkpoint, PREDICTIVE_MODEL_PATH)
    print(f"[PREDICTIVE] Saved model to {PREDICTIVE_MODEL_PATH}")


if __name__ == "__main__":
    main()
