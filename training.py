import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

# ── 1. Load ────────────────────────────────────────────────────────
df = pd.read_csv('Quality of Service 5G.csv')
print("Loaded:", df.shape)

# ── 2. Clean ───────────────────────────────────────────────────────
def extract_number(series):
    return series.str.extract(r'([-\d.]+)').astype(float)

df['Signal_Strength_val']     = extract_number(df['Signal_Strength'])
df['Latency_val']             = extract_number(df['Latency'])
df['Required_Bandwidth_val']  = extract_number(df['Required_Bandwidth'])
df['Allocated_Bandwidth_val'] = extract_number(df['Allocated_Bandwidth'])
df['Resource_Allocation_val'] = extract_number(df['Resource_Allocation'])

# ── 3. Encode ──────────────────────────────────────────────────────
le = LabelEncoder()
df['App_encoded'] = le.fit_transform(df['Application_Type'])
print("Application types found:", le.classes_)

# ── 4. Features and target ─────────────────────────────────────────
X = df[[
    'Signal_Strength_val',
    'Latency_val',
    'Required_Bandwidth_val',
    'Allocated_Bandwidth_val',
    'App_encoded'
]]
y = df['Resource_Allocation_val']

print("\nFeatures shape:", X.shape)
print("Target range:", y.min(), "to", y.max())

# ── 5. Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── 6. Train ───────────────────────────────────────────────────────
print("\nTraining model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# ── 7. Evaluate ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\n=== RESULTS ===")
print(f"MAE  (avg error):  {mae:.2f}%")
print(f"R2   (accuracy):   {r2:.4f}  (1.0 = perfect)")

# ── 8. Overfitting check ───────────────────────────────────────────
train_pred = model.predict(X_train)
train_r2   = r2_score(y_train, train_pred)
test_r2    = r2_score(y_test, y_pred)

print(f"\nTrain R2:   {train_r2:.4f}")
print(f"Test  R2:   {test_r2:.4f}")
print(f"Difference: {train_r2 - test_r2:.4f}")

# ── 9. Cross Validation ────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("\n=== CROSS VALIDATION RESULTS ===")
print(f"Scores per fold:  {cv_scores.round(4)}")
print(f"Average R2:       {cv_scores.mean():.4f}")
print(f"Std deviation:    {cv_scores.std():.4f}")
print(f"Min R2:           {cv_scores.min():.4f}")
print(f"Max R2:           {cv_scores.max():.4f}")

# ── 10. Feature Importance ─────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 5),
                                title='Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nSaved: feature_importance.png")

# ── 11. Predicted vs Actual ────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Resource Allocation %')
plt.ylabel('Predicted Resource Allocation %')
plt.title('Predicted vs Actual')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
print("Saved: predicted_vs_actual.png")

# ── 12. Save model ─────────────────────────────────────────────────
with open('5g_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: 5g_model.pkl")