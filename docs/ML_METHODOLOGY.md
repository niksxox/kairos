# KAIROS — Machine Learning Methodology & Evaluation

**Framework:** Pure NumPy (No `scikit-learn` or black-box estimators)  
**Task:** Hourly Campus Room Request Volume Forecasting  
**Target Variable:** `request_count` (Integer count of incoming requests submitted per hour)  

---

## 1. Problem Formulation

Universities experience acute, cyclic spikes in student study and event space requests (e.g., mid-afternoon peak study periods, pre-exam weeks). Predicting hourly demand allows campus facility administrators to preemptively open overflow rooms, reallocate staff, and prevent student booking rejections.

---

## 2. Model Architecture: Pure NumPy from Scratch

The model is implemented from first principles in `src/ml_model.py`:
- **Hypothesis Function:** $\hat{y} = X w + b$
- **Loss Function:** Mean Squared Error with $L_2$ Ridge Regularization:
  $$J(w, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 + \frac{\lambda}{2m} \|w\|_2^2$$
- **Gradient Optimization:** Batch Gradient Descent:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$
  $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)$$
- **Hyperparameters:** Learning rate $\alpha = 0.05$, Epochs = 2,000, $\lambda_{L2} = 0.01$.

---

## 3. Feature Engineering

10 cyclic and autoregressive features engineered strictly without data leakage:
1. `norm_hour`: Normalized hour of the day $[0, 1]$.
2. `sin_hour`, `cos_hour`: Harmonic cyclical encoding of hour (24-hour period).
3. `sin_weekday`, `cos_weekday`: Harmonic cyclical encoding of day of week (7-day period).
4. `weekday_num`: Categorical integer representation of weekday.
5. `month`: Calendar month of observation.
6. `lag_1`: Request count in the immediately preceding hour.
7. `lag_day`: Request count at the same hour on the previous day (24-hour lag).
8. `rolling_mean_7`: Moving average across the previous 7 hourly periods.

---

## 4. Evaluation Protocol & Chronological Split

To prevent future temporal leakage, data is split chronologically:
- **Training Set (70%):** 1,053 hourly observations.
- **Validation Set (15%):** 226 hourly observations (for early stopping & tuning).
- **Held-Out Test Set (15%):** 226 hourly observations (November 10, 2025 to November 28, 2025).

### Performance vs. Baselines on Held-Out Test Set:
| Model / Baseline | Test MAE | Test RMSE | Improvement vs. Stratified Baseline |
| :--- | :---: | :---: | :---: |
| **Global Training Mean Baseline** | 1.3167 | 1.6573 | Baseline Reference |
| **Same-Weekday-Hour Stratified Mean** | 1.3682 | 1.7278 | Baseline Reference |
| **KAIROS NumPy Linear Regression** | **1.3403** | **1.6892** | **+2.04% MAE / +2.23% RMSE** |

---

## 5. Limitations & Transparency

- The dataset is synthetic and demonstrates Poisson-like discrete demand arrival characteristics.
- Predictions serve as decision-support alerts for campus operations and do **not** automatically override physical room hard constraints or double-booking guards.
