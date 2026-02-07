# Vectionsense

This project implements Transformer-based models to detect and predict cybersickness levels using Inertial Measurement Unit (IMU) data and Heart Rate Variability (HRV) features.

## Prerequisites

Ensure you have the following Python libraries installed:

*   `numpy`
*   `pandas`
*   `torch` (PyTorch)
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `scipy`
*   `moviepy` (optional, for latent alignment videos)

You can install them using pip:

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn scipy moviepy
```

## Usage

To train the model and generate results, run the `trans.py` script:

```bash
python trans.py
```

### Predictive model (future-state prediction)

The predictive pipeline takes a past window `[t-Δ(m), t]` and forecasts the future state at `t+Δ(k)`. It trains a predictive Transformer that outputs a future representation and uses the pretrained `PROPOSED_MODEL` classifier head.

Run:

```bash
python trans_predictive.py
```

Key settings live at the top of `trans_predictive.py`:
*   `LOOKBACK_SEC`, `HORIZON_SEC`, `FUTURE_WINDOW_SEC`, `STRIDE_SEC`
*   `AUX_TARGET` (latent loss by default)
*   `USE_PROPOSED_MODEL` (uses `models_transformer/PROPOSED_MODEL.pth`)

### What the script does:
1.  **Data Loading & Alignment**: Reads IMU and Heart Rate data, aligns them using cubic spline interpolation, and extracts features.
2.  **Feature Extraction**: Calculates velocity, acceleration, jerk, sway variance, and HRV metrics.
3.  **Model Training**: Trains a Time-Series Transformer model on the processed sequences.
4.  **Evaluation**: Performs Leave-One-Out Cross-Validation (LOOCV) or standard evaluation depending on configuration.
5.  **Visualization**: Generates various plots to analyze model performance and feature importance.

## Output

The script generates results in the following directories:

### `plots_transformer/`
Contains detailed visualizations of the model's performance:
*   **Confusion Matrices**: `transformer_confusion_matrix_*.png` (Combined, IMU-only, HR-only).
*   **Feature Importance**: `transformer_feature_importance_top10_*.png` (Top features contributing to predictions).
*   **Learning Curves**: `learning_curve_acc_*.png` and `learning_curve_loss_*.png` (Accuracy and loss over epochs).
*   **Participant Metrics**: `loocv_participantwise_metrics.png` (Accuracy, Precision, Recall, F1-score per participant).
*   **Correlation Plots**: `corr_pearson_sickness_level.png` (Correlation between features and sickness levels).

### `plots_predictive/`
Predictive model diagnostics:
*   **Latent Alignment**: `latent_alignment_epoch_*.png` (PCA comparison of predicted vs true future representations).
*   **Aux Metrics**: `aux_metrics_over_epochs.png`.

### `plots_predictive_eval/`
Predictive model evaluation:
*   **Loss Curve**: `loss_curve.png`.
*   **Accuracy Curve**: `accuracy_curve.png`.
*   **Metrics Bar**: `performance_metrics.png` (Accuracy, Precision, Recall, F1).
*   **Confusion Matrix**: `confusion_matrix.png`.
*   **Video** (optional): `latent_alignment_epochs.mp4` if MoviePy + ffmpeg are available.

## Project Structure

*   `trans.py`: Main script for training and evaluation.
*   `trans_predictive.py`: Predictive pipeline (forecasting future cybersickness).
*   `plots_transformer/`: Generated plots and figures.
*   `results/`: Generated CSV results and summary figures.
*   `hr_aligned/`: Directory containing aligned Heart Rate data.
*   `models_transformer/`: Saved Transformer checkpoints (including `PROPOSED_MODEL.pth`).
