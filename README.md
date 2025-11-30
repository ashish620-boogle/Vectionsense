# Cybersickness Prediction with IMU and HRV

This project implements a Transformer-based model to predict cybersickness levels using Inertial Measurement Unit (IMU) data and Heart Rate Variability (HRV) features.

## Prerequisites

Ensure you have the following Python libraries installed:

*   `numpy`
*   `pandas`
*   `torch` (PyTorch)
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `scipy`

You can install them using pip:

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn scipy
```

## Usage

To train the model and generate results, run the `trans.py` script:

```bash
python trans.py
```

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

### `results/`
Contains summary results and additional comparison plots:
*   **Longevity Analysis**: `longevity_results.csv` and plots showing model performance over time (if applicable).
*   **Model Comparison**: `transformer_models_comparison.png` (Comparison of different model configurations).

## Project Structure

*   `trans.py`: Main script for training and evaluation.
*   `plots_transformer/`: Generated plots and figures.
*   `results/`: Generated CSV results and summary figures.
*   `hr_aligned/`: Directory containing aligned Heart Rate data.
