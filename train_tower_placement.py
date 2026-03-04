"""
Full Pipeline: Handover Gap Detection + Optimal Tower Placement
Dataset : Cellular Network Handover Prediction (India, Airtel 4G/5G)
Outputs :
  - handover_model.pkl   : predicts WHERE handovers will occur (coverage gaps)
  - placement_model.pkl  : predicts quality score at candidate tower positions
  - gap_zones.csv        : lat/lon of detected coverage gap zones
  - tower_recommendations.csv : AI-recommended new tower positions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_absolute_error, r2_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_sample_weight
from scipy.spatial import cKDTree

CSV_PATH = r'network_logs_1.csv'

# ── 1. LOAD & CLEAN ───────────────────────────────────────────────────────────
print("=" * 60)
print("  CELLULAR NETWORK — FULL PIPELINE")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

for col in ['RSRP','RSRQ','SINR','Downlink(Mbps)','Uplink(Mbps)','Velocity(km/h)']:
    df[col] = df[col].astype(str).str.extract(r'([-\d.]+)').astype(float)

df['SINR']     = df['SINR'].clip(-20, 40)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.sort_values(['DeviceID', 'Timestamp']).reset_index(drop=True)

le_prov = LabelEncoder()
df['Provider_enc'] = le_prov.fit_transform(df['Network provi. '].fillna('unknown'))
le_net  = LabelEncoder()
df['NetType_enc']  = le_net.fit_transform(df[' NetworkType'].fillna('unknown'))

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
print("\nEngineering features...")

grp = df.groupby('DeviceID')

# Signal trends
df['RSRP_prev']   = grp['RSRP'].shift(1)
df['RSRP_next']   = grp['RSRP'].shift(-1)
df['RSRP_diff']   = df['RSRP'] - df['RSRP_prev']
df['RSRP_diff2']  = grp['RSRP_diff'].shift(0) - grp['RSRP_diff'].shift(1)   # acceleration
df['RSRP_roll3']  = grp['RSRP'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['RSRP_roll5']  = grp['RSRP'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['RSRP_std3']   = grp['RSRP'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))

df['SINR_prev']   = grp['SINR'].shift(1)
df['SINR_diff']   = df['SINR'] - df['SINR_prev']
df['RSRQ_prev']   = grp['RSRQ'].shift(1)
df['RSRQ_diff']   = df['RSRQ'] - df['RSRQ_prev']

# PCI change flag (handover TARGET)
df['prev_PCI']    = grp['PCI'].shift(1)
df['handover']    = ((df['PCI'] != df['prev_PCI']) & df['prev_PCI'].notna()).astype(int)

# Movement
df['Lat_prev']    = grp['Latitude'].shift(1)
df['Lon_prev']    = grp['Longitude'].shift(1)
df['dist_moved']  = np.sqrt(
    (df['Latitude']  - df['Lat_prev'])**2 +
    (df['Longitude'] - df['Lon_prev'])**2) * 111000   # meters

# Time since last handover
df['time_since_ho'] = df.groupby('DeviceID')['handover'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount())

# PCI stability (same PCI for N consecutive rows)
df['pci_stability'] = df.groupby(['DeviceID','PCI']).cumcount()

# Weak signal flags
df['weak_rsrp']   = (df['RSRP'] < -100).astype(int)
df['poor_sinr']   = (df['SINR'] < -3).astype(int)

# Hour of day
df['hour'] = df['Timestamp'].dt.hour.fillna(12)

print(f"  Handover events    : {df['handover'].sum()} / {len(df)} "
      f"({df['handover'].mean()*100:.2f}%)")
print(f"  Unique towers(PCI) : {df['PCI'].nunique()}")

# ── 3. ESTIMATE REAL TOWER POSITIONS FROM PCI CLUSTERS ───────────────────────
print("\nEstimating real tower positions from PCI signal clusters...")

tower_pos = (df.dropna(subset=['Latitude','Longitude','PCI','RSRP'])
               .groupby('PCI')
               .apply(lambda g: g.nlargest(max(1, len(g)//10), 'RSRP')
                                 [['Latitude','Longitude']].mean())
               .reset_index())
tower_pos.columns = ['PCI','Tower_Lat','Tower_Lon']

# Coverage radius per tower (distance where RSRP > -100 dBm)
def coverage_radius(grp):
    strong = grp[grp['RSRP'] > -100]
    if len(strong) < 3:
        return 500
    lat_c, lon_c = strong['Latitude'].mean(), strong['Longitude'].mean()
    dists = np.sqrt((strong['Latitude']-lat_c)**2 +
                    (strong['Longitude']-lon_c)**2) * 111000
    return float(dists.quantile(0.90))

tower_radius = (df.dropna(subset=['Latitude','Longitude','RSRP'])
                  .groupby('PCI')
                  .apply(coverage_radius)
                  .reset_index())
tower_radius.columns = ['PCI','coverage_radius_m']
tower_pos = tower_pos.merge(tower_radius, on='PCI')

print(f"  Towers estimated   : {len(tower_pos)}")
print(f"  Avg coverage radius: {tower_pos['coverage_radius_m'].mean():.0f} m")

# Merge tower info back
df = df.merge(tower_pos, on='PCI', how='left')
df['dist_to_tower'] = np.sqrt(
    (df['Latitude']  - df['Tower_Lat'])**2 +
    (df['Longitude'] - df['Tower_Lon'])**2) * 111000

# ── 4. MODEL 1: HANDOVER / GAP DETECTION ─────────────────────────────────────
print("\n" + "="*60)
print("  MODEL 1: HANDOVER PREDICTION (Gap Detection)")
print("="*60)

HO_FEATURES = [
    'RSRP', 'RSRQ', 'SINR',
    'RSRP_diff', 'RSRP_diff2', 'RSRP_roll3', 'RSRP_roll5', 'RSRP_std3',
    'SINR_diff', 'RSRQ_diff',
    'dist_moved', 'Velocity(km/h)',
    'dist_to_tower', 'pci_stability', 'time_since_ho',
    'weak_rsrp', 'poor_sinr',
    'Downlink(Mbps)', 'Uplink(Mbps)',
    'Provider_enc', 'NetType_enc', 'hour',
]

df_ho = df.dropna(subset=HO_FEATURES + ['handover']).copy()
X_ho  = df_ho[HO_FEATURES]
y_ho  = df_ho['handover']

print(f"  Samples: {len(X_ho):,}  |  Features: {len(HO_FEATURES)}")
print(f"  Handover rate: {y_ho.mean()*100:.2f}%")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_ho, y_ho, test_size=0.2, random_state=42, stratify=y_ho)

# Handle imbalance with sample weights
sw = compute_sample_weight('balanced', y_tr)

print("\n[1/2] Training Random Forest (handover)...")
rf_ho = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    min_samples_split=4, min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=42)
rf_ho.fit(X_tr, y_tr, sample_weight=sw)

print("[2/2] Training Gradient Boosting (handover)...")
gb_ho = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=5, subsample=0.8,
    min_samples_split=4, random_state=42)
gb_ho.fit(X_tr, y_tr, sample_weight=sw)

# Ensemble probability
ho_proba = (rf_ho.predict_proba(X_te)[:,1] +
            gb_ho.predict_proba(X_te)[:,1]) / 2
ho_pred  = (ho_proba > 0.35).astype(int)   # lower threshold for recall

print("\n  === HANDOVER MODEL RESULTS ===")
print(f"  ROC-AUC  : {roc_auc_score(y_te, ho_proba):.4f}")
print(classification_report(y_te, ho_pred,
                              target_names=['No Handover','Handover'],
                              digits=3))

# Feature importance
imp_ho = pd.Series(rf_ho.feature_importances_, index=HO_FEATURES).sort_values(ascending=False)
print("  Top 5 features:")
print(imp_ho.head(5).to_string())

# ── 5. GAP ZONE DETECTION ─────────────────────────────────────────────────────
print("\n  Detecting coverage gap zones...")

df_ho['ho_proba'] = (rf_ho.predict_proba(X_ho)[:,1] +
                     gb_ho.predict_proba(X_ho)[:,1]) / 2

# Gap zones = locations where handover probability > 0.4
gap_mask   = df_ho['ho_proba'] > 0.40
gap_zones  = df_ho[gap_mask][['Latitude','Longitude','RSRP','SINR','ho_proba']].copy()
gap_zones  = gap_zones.dropna(subset=['Latitude','Longitude'])

# Cluster nearby gaps (within 0.5 km) into single zone
if len(gap_zones) > 0:
    coords   = gap_zones[['Latitude','Longitude']].values
    tree     = cKDTree(coords)
    visited  = np.zeros(len(coords), dtype=bool)
    clusters = []
    for i in range(len(coords)):
        if visited[i]: continue
        nb = tree.query_ball_point(coords[i], r=0.005)   # ~500m
        visited[nb] = True
        cluster_pts = gap_zones.iloc[nb]
        clusters.append({
            'Latitude'  : cluster_pts['Latitude'].mean(),
            'Longitude' : cluster_pts['Longitude'].mean(),
            'avg_RSRP'  : cluster_pts['RSRP'].mean(),
            'avg_SINR'  : cluster_pts['SINR'].mean(),
            'gap_score' : cluster_pts['ho_proba'].mean(),
            'n_events'  : len(nb),
        })
    gap_df = pd.DataFrame(clusters).sort_values('gap_score', ascending=False)
    gap_df.to_csv('gap_zones.csv', index=False)
    print(f"  Gap zones detected : {len(gap_df)}")
    print(f"  Avg gap RSRP       : {gap_df['avg_RSRP'].mean():.1f} dBm")
else:
    gap_df = pd.DataFrame()
    print("  No significant gap zones found.")

# ── 6. MODEL 2: TOWER PLACEMENT QUALITY ───────────────────────────────────────
print("\n" + "="*60)
print("  MODEL 2: TOWER PLACEMENT QUALITY PREDICTOR")
print("="*60)

# Target: signal quality score at each location
# = normalized RSRP + SINR combined (what a new tower should achieve)
df_pl = df.dropna(subset=['Latitude','Longitude','RSRP','SINR',
                           'dist_to_tower','coverage_radius_m']).copy()

# Quality score: how well this location is currently served (0-100)
rsrp_norm  = (df_pl['RSRP'] - df_pl['RSRP'].min()) / (df_pl['RSRP'].max() - df_pl['RSRP'].min())
sinr_norm  = (df_pl['SINR'] - df_pl['SINR'].min()) / (df_pl['SINR'].max() - df_pl['SINR'].min())
dl_norm    = (df_pl['Downlink(Mbps)'].fillna(0) - 0) / (df_pl['Downlink(Mbps)'].max())
df_pl['quality_score'] = (rsrp_norm*0.50 + sinr_norm*0.35 + dl_norm*0.15) * 100

PL_FEATURES = [
    'Latitude', 'Longitude',
    'dist_to_tower', 'coverage_radius_m',
    'RSRP_roll3', 'RSRP_std3',
    'Velocity(km/h)', 'dist_moved',
    'weak_rsrp', 'poor_sinr',
    'Provider_enc', 'NetType_enc', 'hour',
]
df_pl2 = df_pl.dropna(subset=PL_FEATURES + ['quality_score'])
X_pl   = df_pl2[PL_FEATURES]
y_pl   = df_pl2['quality_score']

print(f"  Samples  : {len(X_pl):,}  |  Features: {len(PL_FEATURES)}")
print(f"  Score range: {y_pl.min():.1f} - {y_pl.max():.1f}  mean={y_pl.mean():.1f}")

X_ptr, X_pte, y_ptr, y_pte = train_test_split(X_pl, y_pl, test_size=0.2, random_state=42)

print("\n[1/1] Training placement quality model...")
rf_pl = RandomForestRegressor(
    n_estimators=300, max_depth=12,
    min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_pl.fit(X_ptr, y_ptr)

y_pp = rf_pl.predict(X_pte)
print(f"\n  === PLACEMENT MODEL RESULTS ===")
print(f"  R2   : {r2_score(y_pte, y_pp):.4f}")
print(f"  MAE  : {mean_absolute_error(y_pte, y_pp):.3f} pts")

# ── 7. GENERATE TOWER RECOMMENDATIONS ────────────────────────────────────────
print("\n" + "="*60)
print("  TOWER PLACEMENT RECOMMENDATIONS")
print("="*60)

if len(gap_df) > 0:
    # For each gap zone, predict quality score if we place a tower there
    # Use median stats from existing towers as reference
    med_cov    = tower_pos['coverage_radius_m'].median()
    med_rsrp   = df['RSRP_roll3'].median()
    med_std    = df['RSRP_std3'].median()
    med_vel    = df['Velocity(km/h)'].median()
    med_dist   = 50.0   # new tower placed at gap center = 50m dist

    candidates = []
    for _, row in gap_df.iterrows():
        feat = pd.DataFrame([{
            'Latitude'         : row['Latitude'],
            'Longitude'        : row['Longitude'],
            'dist_to_tower'    : med_dist,
            'coverage_radius_m': med_cov,
            'RSRP_roll3'       : row['avg_RSRP'],
            'RSRP_std3'        : med_std,
            'Velocity(km/h)'   : med_vel,
            'dist_moved'       : 5.0,
            'weak_rsrp'        : int(row['avg_RSRP'] < -100),
            'poor_sinr'        : int(row['avg_SINR'] < -3),
            'Provider_enc'     : 0,
            'NetType_enc'      : 1,
            'hour'             : 12,
        }])
        predicted_quality = rf_pl.predict(feat)[0]
        candidates.append({
            'Rank'            : 0,
            'Latitude'        : round(row['Latitude'], 6),
            'Longitude'       : round(row['Longitude'], 6),
            'Gap_Score'       : round(row['gap_score'], 3),
            'Current_RSRP'    : round(row['avg_RSRP'], 1),
            'Current_SINR'    : round(row['avg_SINR'], 1),
            'Predicted_Quality': round(predicted_quality, 1),
            'Gap_Events'      : row['n_events'],
            'Priority'        : 'HIGH' if row['gap_score'] > 0.6 else
                                'MEDIUM' if row['gap_score'] > 0.45 else 'LOW'
        })

    rec_df = pd.DataFrame(candidates)
    rec_df = rec_df.sort_values('Gap_Score', ascending=False).reset_index(drop=True)
    rec_df['Rank'] = rec_df.index + 1
    rec_df.to_csv('tower_recommendations.csv', index=False)

    print(f"\n  Top 10 recommended tower positions:")
    print(rec_df.head(10)[['Rank','Latitude','Longitude',
                             'Gap_Score','Current_RSRP',
                             'Predicted_Quality','Priority']].to_string(index=False))

# ── 8. PLOTS ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#040a0f')
fig.suptitle('5G Network Gap Detection + Tower Placement Pipeline',
             color='#00e5ff', fontsize=14, y=0.98)

def style_ax(ax, title):
    ax.set_facecolor('#080f18')
    ax.set_title(title, color='#00e5ff', fontsize=10, pad=8)
    ax.tick_params(colors='#cde8f5', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#0d2137')

# 1. Signal map
ax = axes[0,0]
style_ax(ax, 'Signal Strength Map (RSRP)')
sc = ax.scatter(df['Longitude'], df['Latitude'], c=df['RSRP'],
                cmap='RdYlGn', s=2, alpha=0.5, vmin=-130, vmax=-50)
if len(gap_df) > 0:
    ax.scatter(gap_df['Longitude'], gap_df['Latitude'],
               c='red', s=30, marker='x', linewidths=1.5,
               label='Gap zones', zorder=5)
ax.scatter(tower_pos['Tower_Lon'], tower_pos['Tower_Lat'],
           c='cyan', s=15, marker='^', label='Towers', zorder=6)
plt.colorbar(sc, ax=ax, label='RSRP (dBm)').ax.yaxis.label.set_color('#cde8f5')
ax.legend(fontsize=7, facecolor='#040a0f', labelcolor='#cde8f5')
ax.set_xlabel('Longitude', color='#3a5a6e', fontsize=8)
ax.set_ylabel('Latitude', color='#3a5a6e', fontsize=8)

# 2. Handover events map
ax = axes[0,1]
style_ax(ax, 'Handover Events & Gap Probability')
ax.scatter(df['Longitude'], df['Latitude'], c='#1a2a3a', s=1, alpha=0.3)
ho_pts = df[df['handover']==1].dropna(subset=['Latitude','Longitude'])
ax.scatter(ho_pts['Longitude'], ho_pts['Latitude'],
           c='#ff4400', s=20, alpha=0.8, label=f'Handovers ({len(ho_pts)})', zorder=5)
if len(gap_df) > 0:
    sc2 = ax.scatter(gap_df['Longitude'], gap_df['Latitude'],
                     c=gap_df['gap_score'], cmap='hot',
                     s=gap_df['n_events']*2+10, alpha=0.9,
                     label='Gap zones', zorder=6)
    plt.colorbar(sc2, ax=ax, label='Gap Score').ax.yaxis.label.set_color('#cde8f5')
ax.legend(fontsize=7, facecolor='#040a0f', labelcolor='#cde8f5')
ax.set_xlabel('Longitude', color='#3a5a6e', fontsize=8)

# 3. Feature importance (handover model)
ax = axes[0,2]
style_ax(ax, 'Top Features — Handover Model')
imp_ho.head(10).sort_values().plot(kind='barh', ax=ax, color='#00e5ff')
ax.set_xlabel('Importance', color='#3a5a6e', fontsize=8)

# 4. ROC-like: handover probability distribution
ax = axes[1,0]
style_ax(ax, 'Handover Probability Distribution')
df_ho['ho_proba_plot'] = (rf_ho.predict_proba(X_ho)[:,1] +
                           gb_ho.predict_proba(X_ho)[:,1]) / 2
ax.hist(df_ho[df_ho['handover']==0]['ho_proba_plot'],
        bins=40, alpha=0.6, color='#00e5ff', label='No Handover', density=True)
ax.hist(df_ho[df_ho['handover']==1]['ho_proba_plot'],
        bins=40, alpha=0.7, color='#ff4400', label='Handover', density=True)
ax.axvline(0.35, color='#ffee00', linestyle='--', lw=1.5, label='Threshold 0.35')
ax.legend(fontsize=7, facecolor='#040a0f', labelcolor='#cde8f5')
ax.set_xlabel('Predicted Probability', color='#3a5a6e', fontsize=8)

# 5. Tower recommendations map
ax = axes[1,1]
style_ax(ax, 'Recommended New Tower Positions')
ax.scatter(df['Longitude'], df['Latitude'], c='#1a2a3a', s=1, alpha=0.2)
ax.scatter(tower_pos['Tower_Lon'], tower_pos['Tower_Lat'],
           c='#00e5ff', s=20, marker='^', label='Existing towers', zorder=5)
if len(gap_df) > 0 and 'rec_df' in dir():
    colors = {'HIGH':'#ff2244','MEDIUM':'#ff9900','LOW':'#ffee00'}
    for pri, col in colors.items():
        sub = rec_df[rec_df['Priority']==pri]
        if len(sub):
            ax.scatter(sub['Longitude'], sub['Latitude'],
                       c=col, s=60, marker='*', label=f'{pri} priority', zorder=6)
ax.legend(fontsize=7, facecolor='#040a0f', labelcolor='#cde8f5')
ax.set_xlabel('Longitude', color='#3a5a6e', fontsize=8)

# 6. Quality score distribution
ax = axes[1,2]
style_ax(ax, 'Predicted Quality at Recommended Sites')
if 'rec_df' in dir() and len(rec_df) > 0:
    colors_bar = ['#ff2244' if p=='HIGH' else '#ff9900' if p=='MEDIUM'
                  else '#ffee00' for p in rec_df.head(15)['Priority']]
    ax.barh(range(min(15,len(rec_df))),
            rec_df.head(15)['Predicted_Quality'], color=colors_bar)
    ax.set_yticks(range(min(15,len(rec_df))))
    ax.set_yticklabels([f"Site {r}" for r in rec_df.head(15)['Rank']],
                        fontsize=7, color='#cde8f5')
    ax.set_xlabel('Predicted Quality Score (0-100)', color='#3a5a6e', fontsize=8)
    ax.axvline(50, color='#00e5ff', lw=1, linestyle='--')

plt.tight_layout()
plt.savefig('pipeline_results.png', dpi=150, facecolor='#040a0f')
print("\nSaved: pipeline_results.png")

# ── 9. SAVE MODELS ────────────────────────────────────────────────────────────
with open('handover_model.pkl', 'wb') as f:
    pickle.dump({
        'rf': rf_ho, 'gb': gb_ho,
        'features': HO_FEATURES,
        'threshold': 0.35,
        'tower_positions': tower_pos,
    }, f)

with open('placement_model.pkl', 'wb') as f:
    pickle.dump({
        'rf': rf_pl,
        'features': PL_FEATURES,
        'tower_positions': tower_pos,
        'gap_zones': gap_df if len(gap_df) > 0 else pd.DataFrame(),
        'recommendations': rec_df if 'rec_df' in dir() else pd.DataFrame(),
    }, f)

print("Saved: handover_model.pkl")
print("Saved: placement_model.pkl")
print("Saved: gap_zones.csv")
print("Saved: tower_recommendations.csv")
print("\nPipeline complete.")
