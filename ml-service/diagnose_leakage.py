"""
diagnose_leakage.py
====================
Investigates whether features like average_usage / maximum_usage / task_duration
are causing data leakage (i.e., they encode information about the outcome).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

DATA_PATH = '../archive/borg_traces_data.csv'
RANDOM_STATE = 42

print("Loading data...")
df = pd.read_csv(DATA_PATH)

def parse_array(s):
    try:
        return np.fromstring(str(s).strip('[]'), sep=' ')
    except Exception:
        return np.array([np.nan])

def extract_struct_field(s, field):
    try:
        import ast
        d = ast.literal_eval(str(s))
        return float(d.get(field, np.nan))
    except Exception:
        return np.nan

# Parse string columns
cpu_arrays = df['cpu_usage_distribution'].apply(parse_array)
n_bins = int(cpu_arrays.dropna().iloc[0].shape[0])
for i in range(n_bins):
    df[f'cpu_dist_p{i}'] = cpu_arrays.apply(lambda x, i=i: x[i] if len(x) > i else np.nan)
df['cpu_dist_mean'] = cpu_arrays.apply(lambda x: x.mean() if not np.isnan(x).all() else np.nan)
df['cpu_dist_max']  = cpu_arrays.apply(lambda x: x.max()  if not np.isnan(x).all() else np.nan)
df['cpu_dist_std']  = cpu_arrays.apply(lambda x: x.std()  if not np.isnan(x).all() else np.nan)
df['cpu_dist_p95']  = cpu_arrays.apply(lambda x: np.percentile(x, 95) if not np.isnan(x).all() else np.nan)

if 'tail_cpu_usage_distribution' in df.columns:
    tail = df['tail_cpu_usage_distribution'].apply(parse_array)
    df['tail_cpu_dist_mean'] = tail.apply(lambda x: x.mean() if not np.isnan(x).all() else np.nan)
    df['tail_cpu_dist_max']  = tail.apply(lambda x: x.max()  if not np.isnan(x).all() else np.nan)
    df['tail_cpu_dist_p95']  = tail.apply(lambda x: np.percentile(x, 95) if not np.isnan(x).all() else np.nan)

for col_name, prefix in [('average_usage', 'avg'), ('maximum_usage', 'max_u'), ('random_sample_usage', 'sample')]:
    if col_name in df.columns:
        df[f'{prefix}_cpu']    = df[col_name].apply(lambda s: extract_struct_field(s, 'cpus'))
        df[f'{prefix}_memory'] = df[col_name].apply(lambda s: extract_struct_field(s, 'memory'))

if 'start_time' in df.columns and 'end_time' in df.columns:
    df['task_duration'] = df['end_time'] - df['start_time']

# Sort, temporal features, impute
df = df.sort_values(['machine_id', 'time']).reset_index(drop=True)
df['cpu_lag1']      = df.groupby('machine_id')['cpu_dist_mean'].shift(1)
df['cpu_lag2']      = df.groupby('machine_id')['cpu_dist_mean'].shift(2)
df['cpu_roll_mean'] = df.groupby('machine_id')['cpu_dist_mean'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean())
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

split = int(len(df) * 0.8)
target = 'failed'

# ── Feature sets ──────────────────────────────────────────────────────────────

# Set A: ALL features (original, potentially leaky)
dist_features = [f'cpu_dist_p{i}' for i in range(n_bins)]
FEATURES_ALL = (
    dist_features +
    ['cpu_dist_mean','cpu_dist_max','cpu_dist_std','cpu_dist_p95'] +
    ['cpu_lag1','cpu_lag2','cpu_roll_mean'] +
    ['assigned_memory','scheduling_class','priority'] +
    [f for f in ['avg_cpu','avg_memory','max_u_cpu','max_u_memory',
                 'sample_cpu','sample_memory',
                 'tail_cpu_dist_mean','tail_cpu_dist_max','tail_cpu_dist_p95',
                 'task_duration'] if f in df.columns]
)

# Set B: SAFE features only — available in real-time BEFORE the task ends
# Removed: average_usage, maximum_usage, random_sample_usage, task_duration
# These are computed over the ENTIRE task lifetime → look-ahead leakage
FEATURES_SAFE = (
    dist_features +
    ['cpu_dist_mean','cpu_dist_max','cpu_dist_std','cpu_dist_p95'] +
    ['cpu_lag1','cpu_lag2','cpu_roll_mean'] +
    ['assigned_memory','scheduling_class','priority'] +
    [f for f in ['tail_cpu_dist_mean','tail_cpu_dist_max','tail_cpu_dist_p95']
     if f in df.columns]
)

print(f"\nFeatures ALL : {len(FEATURES_ALL)}")
print(f"Features SAFE: {len(FEATURES_SAFE)}")
print("\nSuspect (REMOVED from SAFE set):")
removed = set(FEATURES_ALL) - set(FEATURES_SAFE)
for f in sorted(removed):
    # Show how much the means differ between failed and not-failed
    if f in df.columns:
        m0 = df.loc[df[target]==0, f].mean()
        m1 = df.loc[df[target]==1, f].mean()
        sep = abs(m1 - m0) / (df[f].std() + 1e-9)
        print(f"  {f:30s}  mean(ok)={m0:.4f}  mean(fail)={m1:.4f}  separation={sep:.2f}σ")

# ── Train & compare ───────────────────────────────────────────────────────────
results = {}
for label, feats in [("ALL (leaky?)", FEATURES_ALL), ("SAFE (clean)", FEATURES_SAFE)]:
    feats = [f for f in feats if f in df.columns]
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    X_tr, y_tr = train_df[feats].values, train_df[target].values
    X_te, y_te = test_df[feats].values,  test_df[target].values

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=10,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    results[label] = (auc, rf, feats, sc)
    print(f"\n[{label}]  ROC-AUC = {auc:.4f}")
    print(classification_report(y_te, (proba >= 0.5).astype(int),
                                target_names=['not_failed','failed']))

# ── Feature importances (safe model) ─────────────────────────────────────────
_, best_rf, best_feats, best_sc = results["SAFE (clean)"]
imps = pd.Series(best_rf.feature_importances_, index=best_feats).sort_values(ascending=False)
print("\nTop 15 feature importances (SAFE model):")
print(imps.head(15).to_string())

# ── Save clean model ──────────────────────────────────────────────────────────
auc_all  = results["ALL (leaky?)"][0]
auc_safe = results["SAFE (clean)"][0]
print(f"\n{'='*50}")
print(f"ALL features AUC  : {auc_all:.4f}")
print(f"SAFE features AUC : {auc_safe:.4f}")
drop = auc_all - auc_safe
if drop > 0.05:
    print(f"*** AUC dropped by {drop:.4f} → CONFIRMED DATA LEAKAGE in removed features ***")
    print("Saving the SAFE model as the production model...")
    joblib.dump(best_rf, 'failure_model.pkl')
    joblib.dump(best_sc,  'scaler.pkl')
    # Save clean feature list so API can use it
    import json
    with open('model_features.json','w') as f:
        json.dump(best_feats, f, indent=2)
    print("Saved: failure_model.pkl, scaler.pkl, model_features.json")
else:
    print(f"AUC drop only {drop:.4f} — leakage impact is minimal.")
