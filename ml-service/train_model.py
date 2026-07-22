"""
train_model.py — Cloud Resource Failure Prediction
====================================================
Binary classification: predict whether a task/machine will fail (failed=1).

Pipeline
--------
1. Load raw data
2. Parse string-array columns → numeric features
3. Sort by time, compute lag/rolling features on numeric column
4. Impute missing values (median)
5. Train RandomForest (+ LightGBM if installed)
6. Evaluate with classification metrics
7. Save best model with joblib
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = '../archive/borg_traces_data.csv'
MODEL_PATH = 'failure_model.pkl'
SCALER_PATH = 'scaler.pkl'
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df.shape}")


# ── 2. Parse string-array columns into numeric features ───────────────────────
print("Parsing string-array columns...")

def parse_array(s):
    """Parse '[0.1 0.2 ...]' → numpy array."""
    try:
        return np.fromstring(str(s).strip('[]'), sep=' ')
    except Exception:
        return np.array([np.nan])

def extract_struct_field(s, field):
    """Parse dict-like string and extract a float field."""
    try:
        import ast
        d = ast.literal_eval(str(s))
        return float(d.get(field, np.nan))
    except Exception:
        return np.nan

# cpu_usage_distribution → 11 percentile bins + summary stats
cpu_arrays = df['cpu_usage_distribution'].apply(parse_array)
n_bins = int(cpu_arrays.dropna().iloc[0].shape[0])
for i in range(n_bins):
    df[f'cpu_dist_p{i}'] = cpu_arrays.apply(lambda x, i=i: x[i] if len(x) > i else np.nan)
df['cpu_dist_mean'] = cpu_arrays.apply(lambda x: x.mean() if not np.isnan(x).all() else np.nan)
df['cpu_dist_max']  = cpu_arrays.apply(lambda x: x.max()  if not np.isnan(x).all() else np.nan)
df['cpu_dist_std']  = cpu_arrays.apply(lambda x: x.std()  if not np.isnan(x).all() else np.nan)
df['cpu_dist_p95']  = cpu_arrays.apply(lambda x: np.percentile(x, 95) if not np.isnan(x).all() else np.nan)

# tail_cpu_usage_distribution
if 'tail_cpu_usage_distribution' in df.columns:
    tail = df['tail_cpu_usage_distribution'].apply(parse_array)
    df['tail_cpu_dist_mean'] = tail.apply(lambda x: x.mean() if not np.isnan(x).all() else np.nan)
    df['tail_cpu_dist_max']  = tail.apply(lambda x: x.max()  if not np.isnan(x).all() else np.nan)
    df['tail_cpu_dist_p95']  = tail.apply(lambda x: np.percentile(x, 95) if not np.isnan(x).all() else np.nan)

# Struct columns: average_usage, maximum_usage, random_sample_usage
for col_name, prefix in [('average_usage', 'avg'), ('maximum_usage', 'max_u'),
                          ('random_sample_usage', 'sample')]:
    if col_name in df.columns:
        df[f'{prefix}_cpu']    = df[col_name].apply(lambda s: extract_struct_field(s, 'cpus'))
        df[f'{prefix}_memory'] = df[col_name].apply(lambda s: extract_struct_field(s, 'memory'))

# Task duration
if 'start_time' in df.columns and 'end_time' in df.columns:
    df['task_duration'] = df['end_time'] - df['start_time']

print(f"  Columns after parsing: {df.shape[1]}")


# ── 3. Sort by time, compute temporal features ────────────────────────────────
print("Computing temporal features...")
df = df.sort_values(['machine_id', 'time']).reset_index(drop=True)

df['cpu_lag1']      = df.groupby('machine_id')['cpu_dist_mean'].shift(1)
df['cpu_lag2']      = df.groupby('machine_id')['cpu_dist_mean'].shift(2)
df['cpu_roll_mean'] = (
    df.groupby('machine_id')['cpu_dist_mean']
      .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)


# ── 4. Impute missing values ──────────────────────────────────────────────────
print("Imputing missing values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"  Shape after imputation: {df.shape}")


# ── 5. Feature selection & train/test split ───────────────────────────────────
dist_features = [f'cpu_dist_p{i}' for i in range(n_bins)]
stat_features = ['cpu_dist_mean', 'cpu_dist_max', 'cpu_dist_std', 'cpu_dist_p95']
temp_features = ['cpu_lag1', 'cpu_lag2', 'cpu_roll_mean']
base_features = ['assigned_memory', 'scheduling_class', 'priority']
optional      = ['avg_cpu', 'avg_memory', 'max_u_cpu', 'max_u_memory',
                 'sample_cpu', 'sample_memory',
                 'tail_cpu_dist_mean', 'tail_cpu_dist_max', 'tail_cpu_dist_p95',
                 'task_duration']

features = dist_features + stat_features + temp_features + base_features
features += [f for f in optional if f in df.columns]
features  = [f for f in features if f in df.columns]
target    = 'failed'

print(f"  Features used: {len(features)}")
print(f"  Class distribution:\n{df[target].value_counts(normalize=True).round(3)}")

# Time-based split (not random — respects temporal order)
split = int(len(df) * TRAIN_RATIO)
train_df = df.iloc[:split]
test_df  = df.iloc[split:]

X_train, y_train = train_df[features].values, train_df[target].values
X_test,  y_test  = test_df[features].values,  test_df[target].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, SCALER_PATH)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")


# ── 6. Train models ───────────────────────────────────────────────────────────
results = {}

# — RandomForest —
print("\nTraining RandomForest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=10,
    class_weight='balanced',   # handles 77/23 imbalance
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
results['RandomForest'] = rf

# — LightGBM (if installed) —
try:
    import lightgbm as lgb
    print("Training LightGBM...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(100)])
    results['LightGBM'] = lgbm
except ImportError:
    print("  LightGBM not installed — skipping (pip install lightgbm to enable).")


# ── 7. Evaluate all models ────────────────────────────────────────────────────
best_name, best_model, best_auc = None, None, -1

for name, model in results.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['not_failed', 'failed']))

    if auc > best_auc:
        best_auc, best_name, best_model = auc, name, model

# ── 8. Save best model ────────────────────────────────────────────────────────

print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")
joblib.dump(best_model, MODEL_PATH)
print(f"Saved → {MODEL_PATH}  |  Scaler → {SCALER_PATH}")
print("\nFeatures used:")
for f in features:
    print(f"  {f}")

# ── 9. Visual Performance Summary ──────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import RocCurveDisplay

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, target_names=['not_failed', 'failed'], output_dict=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Model Performance — {best_name}", fontsize=16, fontweight='bold')

    # ROC Curve
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axs[0], name=f"ROC-AUC = {auc:.4f}")
    axs[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axs[0].set_title("ROC Curve")
    axs[0].legend(loc='lower right')

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1], cbar=False,
                xticklabels=['Not Failed', 'Failed'], yticklabels=['Not Failed', 'Failed'])
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Actual')
    axs[1].set_title('Confusion Matrix')

    # Metrics box
    metrics_text = (
        f"Accuracy: {cr['accuracy']:.4f}\n"
        f"F1 (fail): {cr['failed']['f1-score']:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n"
        f"Avg Precision: {ap:.4f}"
    )
    plt.gcf().text(0.75, 0.25, metrics_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
except Exception as e:
    print(f"[WARN] Could not display performance plot: {e}")
