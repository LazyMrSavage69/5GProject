
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
df = pd.read_csv('signal_metrics.csv')
print(f"Loaded full dataset : {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── 2. FILTER 5G ONLY ──────────────────────────────────────────────────────────
df = df[df['Network Type'] == '5G'].copy()
print(f"5G rows kept        : {len(df):,}")

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
df['Avg_SDR_dBm'] = df[['BB60C Measurement (dBm)',
                          'srsRAN Measurement (dBm)',
                          'BladeRFxA9 Measurement (dBm)']].mean(axis=1)

df['SDR_Spread_dBm'] = (
    df[['BB60C Measurement (dBm)',
        'srsRAN Measurement (dBm)',
        'BladeRFxA9 Measurement (dBm)']].max(axis=1) -
    df[['BB60C Measurement (dBm)',
        'srsRAN Measurement (dBm)',
        'BladeRFxA9 Measurement (dBm)']].min(axis=1)
)

df['Signal_vs_SDR']  = df['Signal Strength (dBm)'] - df['Avg_SDR_dBm']
df['BB60C_vs_srsRAN'] = df['BB60C Measurement (dBm)'] - df['srsRAN Measurement (dBm)']

le = LabelEncoder()
df['Locality_enc'] = le.fit_transform(df['Locality'])
print(f"Localities ({df['Locality'].nunique()})     : {list(le.classes_)}")

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# ── 4. BUILD PHYSICS-BASED TARGET ─────────────────────────────────────────────
# Signal Quality Score (0-100): derived from 5G NR domain knowledge
# Higher signal + lower spread + consistent SDR readings = better quality
sig_norm    = (df['Signal Strength (dBm)'] - df['Signal Strength (dBm)'].min()) / \
              (df['Signal Strength (dBm)'].max() - df['Signal Strength (dBm)'].min())
sdr_norm    = (df['Avg_SDR_dBm'] - df['Avg_SDR_dBm'].min()) / \
              (df['Avg_SDR_dBm'].max() - df['Avg_SDR_dBm'].min())
spread_norm = 1 - (df['SDR_Spread_dBm'] - df['SDR_Spread_dBm'].min()) / \
              (df['SDR_Spread_dBm'].max() - df['SDR_Spread_dBm'].min())

df['Signal_Quality_Score'] = (sig_norm * 0.40 + sdr_norm * 0.35 + spread_norm * 0.25) * 100

print(f"\nTarget (Signal Quality Score):")
print(f"  Range : {df['Signal_Quality_Score'].min():.2f} -> {df['Signal_Quality_Score'].max():.2f}")
print(f"  Mean  : {df['Signal_Quality_Score'].mean():.2f} | Std: {df['Signal_Quality_Score'].std():.2f}")

# ── 5. FEATURES & TARGET ──────────────────────────────────────────────────────
FEATURES = [
    'Signal Strength (dBm)',
    'Latency (ms)',
    'Data Throughput (Mbps)',
    'BB60C Measurement (dBm)',
    'srsRAN Measurement (dBm)',
    'BladeRFxA9 Measurement (dBm)',
    'Avg_SDR_dBm',
    'SDR_Spread_dBm',
    'Signal_vs_SDR',
    'BB60C_vs_srsRAN',
    'Locality_enc',
    'Latitude',
    'Longitude',
    'Hour',
]
TARGET = 'Signal_Quality_Score'
X = df[FEATURES]
y = df[TARGET]
print(f"\nFeatures: {len(FEATURES)} | Samples: {len(X):,}")

# ── 6. SPLIT ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 7. TRAIN ──────────────────────────────────────────────────────────────────
print("\n[1/2] Training Random Forest (n=300, depth=12)...")
rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4,
                            min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

print("[2/2] Training Gradient Boosting (n=300, lr=0.05)...")
gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                min_samples_split=4, min_samples_leaf=2,
                                subsample=0.8, random_state=42)
gb.fit(X_train, y_train)

def ensemble_predict(X_inp):
    return (rf.predict(X_inp) + gb.predict(X_inp)) / 2

# ── 8. EVALUATE ───────────────────────────────────────────────────────────────
print("\n" + "="*54)
print("  MODEL EVALUATION")
print("="*54)
for name, pred_fn in [("Random Forest", rf.predict),
                       ("Gradient Boosting", gb.predict),
                       ("Ensemble (avg)", ensemble_predict)]:
    yp = pred_fn(X_test)
    print(f"\n  {name}")
    print(f"    MAE  : {mean_absolute_error(y_test, yp):.4f} score pts")
    print(f"    RMSE : {mean_squared_error(y_test, yp)**0.5:.4f} score pts")
    print(f"    R2   : {r2_score(y_test, yp):.4f}")

# ── 9. OVERFITTING CHECK ──────────────────────────────────────────────────────
print("\n" + "="*54)
print("  OVERFITTING CHECK - Random Forest")
print("="*54)
tr2 = r2_score(y_train, rf.predict(X_train))
te2 = r2_score(y_test,  rf.predict(X_test))
print(f"  Train R2 : {tr2:.4f}")
print(f"  Test  R2 : {te2:.4f}")
print(f"  Gap      : {tr2-te2:.4f}  {'OK' if tr2-te2 < 0.05 else 'slight overfit - expected'}")

# ── 10. CROSS VALIDATION ──────────────────────────────────────────────────────
print("\n" + "="*54)
print("  5-FOLD CROSS VALIDATION - Random Forest")
print("="*54)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2  = cross_val_score(rf, X, y, cv=kf, scoring='r2', n_jobs=-1)
cv_mae = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
print(f"  R2 per fold : {cv_r2.round(4)}")
print(f"  R2 mean     : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  MAE mean    : {(-cv_mae).mean():.4f} +/- {(-cv_mae).std():.4f} pts")

# ── 11. PLOTS ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#040a0f')

ax = axes[0]
ax.set_facecolor('#080f18')
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
colors = ['#39ff14' if i >= len(imp)-3 else '#00e5ff' if i >= len(imp)-6
          else '#1a4a5e' for i in range(len(imp))]
imp.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Importance (RF)', color='#00e5ff', fontsize=11)
ax.tick_params(colors='#cde8f5', labelsize=7)
ax.set_xlabel('Importance', color='#3a5a6e')
for sp in ax.spines.values(): sp.set_edgecolor('#0d2137')

ax2 = axes[1]
ax2.set_facecolor('#080f18')
yp_ens = ensemble_predict(X_test)
ax2.scatter(y_test, yp_ens, alpha=0.3, color='#00e5ff', s=6)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
ax2.set_xlabel('Actual Score', color='#3a5a6e')
ax2.set_ylabel('Predicted Score', color='#3a5a6e')
ax2.set_title('Predicted vs Actual - Ensemble', color='#00e5ff', fontsize=11)
ax2.tick_params(colors='#cde8f5', labelsize=8)
for sp in ax2.spines.values(): sp.set_edgecolor('#0d2137')

ax3 = axes[2]
ax3.set_facecolor('#080f18')
loc_scores = df.groupby('Locality')['Signal_Quality_Score'].mean().sort_values()
colors3 = ['#39ff14' if v > 70 else '#00e5ff' if v > 50 else '#ff6b35'
           for v in loc_scores.values]
loc_scores.plot(kind='barh', ax=ax3, color=colors3)
ax3.set_title('Avg Signal Quality by Locality', color='#00e5ff', fontsize=11)
ax3.set_xlabel('Score (0-100)', color='#3a5a6e')
ax3.tick_params(colors='#cde8f5', labelsize=7)
for sp in ax3.spines.values(): sp.set_edgecolor('#0d2137')

plt.tight_layout()
plt.savefig('5g_model_results.png', dpi=150, facecolor='#040a0f')
print("\nSaved: 5g_model_results.png")

# ── 12. SAVE ARTIFACTS ────────────────────────────────────────────────────────
artifacts = {
    'rf_model':      rf,
    'gb_model':      gb,
    'label_encoder': le,
    'feature_names': FEATURES,
    'target':        TARGET,
    'target_range':  (float(y.min()), float(y.max())),
    'localities':    list(le.classes_),
}
with open('5g_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print("Saved: 5g_model.pkl")

# ── 13. SAMPLE INFERENCE ──────────────────────────────────────────────────────
print("\n" + "="*54)
print("  SAMPLE PREDICTION")
print("="*54)
sample = X_test.iloc[[0]]
rf_p   = rf.predict(sample)[0]
gb_p   = gb.predict(sample)[0]
ens_p  = (rf_p + gb_p) / 2
actual = y_test.iloc[0]
loc_name = le.inverse_transform([int(sample['Locality_enc'].values[0])])[0]
print(f"  Locality : {loc_name}")
print(f"  Actual   : {actual:.2f} / 100")
print(f"  RF pred  : {rf_p:.2f} / 100")
print(f"  GB pred  : {gb_p:.2f} / 100")
print(f"  Ensemble : {ens_p:.2f} / 100")
print(f"  Error    : {abs(ens_p - actual):.4f} pts")
print("\nTraining complete.")
