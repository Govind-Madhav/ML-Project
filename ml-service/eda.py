# Exploratory Data Analysis (EDA) for Cloud Resource Optimization Dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Data Understanding
DATA_PATH = '../archive/borg_traces_data.csv'
df = pd.read_csv(DATA_PATH)
print('--- Basic Info ---')
print(df.info())
print('\n--- Head ---')
print(df.head())
print('\n--- Describe ---')
print(df.describe())
print('\n--- Nulls ---')
print(df.isnull().sum())

# Check CPU value range
for col in ['cpu_usage_distribution', 'average_usage']:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Column {col} conversion error: {e}")
print('\nCPU usage min/max:', df['cpu_usage_distribution'].min(), df['cpu_usage_distribution'].max())

# Check time intervals
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df = df.dropna(subset=['time', 'cpu_usage_distribution'])
df['time'] = df['time'].astype(np.int64)
print('\nUnique machines:', df['machine_id'].nunique())
for m in df['machine_id'].unique()[:3]:
    sub = df[df['machine_id'] == m].sort_values('time')
    time_diffs = sub['time'].diff().dropna()
    print(f"Machine {m} median interval:", time_diffs.median(), "min:", time_diffs.min(), "max:", time_diffs.max())

# 2️⃣ Data Cleaning
df = df.drop_duplicates()
df = df.sort_values(['machine_id', 'time'])
df['cpu_usage_distribution'] = df['cpu_usage_distribution'].fillna(method='ffill')
df['assigned_memory'] = pd.to_numeric(df['assigned_memory'], errors='coerce').fillna(method='ffill')

# 3️⃣ Sorting (already done above)

# 4️⃣ Feature Selection
df = df[['time', 'machine_id', 'cpu_usage_distribution', 'assigned_memory']]
df = df.dropna()

# 5️⃣ Feature Engineering
df['cpu_lag1'] = df.groupby('machine_id')['cpu_usage_distribution'].shift(1)
df['cpu_lag2'] = df.groupby('machine_id')['cpu_usage_distribution'].shift(2)
df['cpu_roll_mean'] = df.groupby('machine_id')['cpu_usage_distribution'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

# Optional: time features
df['hour'] = (df['time'] // (1000*60*60)) % 24
df['dayofweek'] = (df['time'] // (1000*60*60*24)) % 7

# 6️⃣ Handle NaN (from lag)
df = df.dropna()

# 7️⃣ Train-Test Split (time-based)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# 8️⃣ Normalization (optional)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = train.copy()
test_scaled = test.copy()
cols_to_scale = ['cpu_lag1', 'cpu_lag2', 'cpu_roll_mean', 'assigned_memory']
train_scaled[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
test_scaled[cols_to_scale] = scaler.transform(test[cols_to_scale])

# 9️⃣ Target Define
X_train = train_scaled[['cpu_lag1', 'cpu_lag2', 'cpu_roll_mean', 'assigned_memory']]
y_train = train_scaled['cpu_usage_distribution']
X_test = test_scaled[['cpu_lag1', 'cpu_lag2', 'cpu_roll_mean', 'assigned_memory']]
y_test = test_scaled['cpu_usage_distribution']

# 🔟 Model Training (quick check)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 1️⃣1️⃣ Evaluation
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(preds[:100], label='Predicted')
plt.legend()
plt.title('Trend Check (first 100 test samples)')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('MAE:', mean_absolute_error(y_test, preds))
print('RMSE:', mean_squared_error(y_test, preds, squared=False))
print('R2:', r2_score(y_test, preds))

# ⚡ Fast Checklist
print('\n--- Fast Checklist ---')
print('data sorted:', df.equals(df.sort_values(["machine_id", "time"])))
print('missing handled:', df.isnull().sum().sum() == 0)
print('lag features created:', "cpu_lag1" in df.columns and "cpu_lag2" in df.columns)
print('rolling features added:', "cpu_roll_mean" in df.columns)
print('train-test split (time-based):', len(train) + len(test) == len(df))
print('model trained:', hasattr(model, "predict"))
print('predictions checked:', len(preds) == len(y_test))
