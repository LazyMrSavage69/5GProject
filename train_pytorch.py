import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

# ── 1. Load and Clean ──────────────────────────────────────────────
df = pd.read_csv('Quality of Service 5G.csv')
print("Loaded:", df.shape)

def extract_number(series):
    return series.str.extract(r'([-\d.]+)').astype(float)

df['Signal_Strength_val']     = extract_number(df['Signal_Strength'])
df['Latency_val']             = extract_number(df['Latency'])
df['Required_Bandwidth_val']  = extract_number(df['Required_Bandwidth'])
df['Allocated_Bandwidth_val'] = extract_number(df['Allocated_Bandwidth'])
df['Resource_Allocation_val'] = extract_number(df['Resource_Allocation'])

le = LabelEncoder()
df['App_encoded'] = le.fit_transform(df['Application_Type'])

# ── 2. Features and Target ─────────────────────────────────────────
X = df[[
    'Signal_Strength_val',
    'Latency_val',
    'Required_Bandwidth_val',
    'Allocated_Bandwidth_val',
    'App_encoded'
]].values

y = df['Resource_Allocation_val'].values

# ── 3. Scale ───────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# ── 4. Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── 5. Dataset Class ───────────────────────────────────────────────
class NetworkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NetworkDataset(X_train, y_train)
test_dataset  = NetworkDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

# ── 6. Improved Architecture ───────────────────────────────────────
class Network5G(nn.Module):
    def __init__(self):
        super(Network5G, self).__init__()
        self.model = nn.Sequential(
            # Block 1
            nn.Linear(5, 128),
            nn.BatchNorm1d(128),     # stabilizes training
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 3
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Block 4
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()

# ── 7. Setup ───────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model     = Network5G().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

# Reduce LR when stuck
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)
# ── 8. Early Stopping Class ────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.should_stop = False
        self.best_weights = None

    def check(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# ── 9. Training Loop ───────────────────────────────────────────────
print("\nTraining optimized neural network...")
epochs        = 500
train_losses  = []
test_losses   = []
early_stop    = EarlyStopping(patience=30)

for epoch in range(epochs):
    # Train
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        batch_losses.append(loss.item())

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_batch_losses = []
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)
    test_loss  = np.mean(test_batch_losses)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    scheduler.step(test_loss)
    early_stop.check(test_loss, model)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if early_stop.should_stop:
        print(f"\nEarly stopping at epoch {epoch+1}")
        model.load_state_dict(early_stop.best_weights)
        break

# ── 10. Final Evaluation ───────────────────────────────────────────
model.eval()
with torch.no_grad():
    X_test_tensor  = torch.FloatTensor(X_test).to(device)
    y_pred_scaled  = model(X_test_tensor).cpu().numpy()

y_pred      = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1,  1)).flatten()

mae = mean_absolute_error(y_test_orig, y_pred)
r2  = r2_score(y_test_orig, y_pred)

print(f"\n=== OPTIMIZED PYTORCH RESULTS ===")
print(f"MAE  (avg error):  {mae:.2f}%")
print(f"R2   (accuracy):   {r2:.4f}  (1.0 = perfect)")

print(f"\n=== COMPARISON ===")
print(f"Random Forest R2:       0.8966")
print(f"PyTorch basic R2:       0.9261")
print(f"PyTorch optimized R2:   {r2:.4f}")

# ── 11. Plot Training Loss ─────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss', color='steelblue')
plt.plot(test_losses,  label='Test Loss',  color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.tight_layout()
plt.savefig('training_loss_optimized.png')
print("\nSaved: training_loss_optimized.png")

# ── 12. Predicted vs Actual ────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(y_test_orig, y_pred, alpha=0.6, color='steelblue')
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()], 'r--')
plt.xlabel('Actual Resource Allocation %')
plt.ylabel('Predicted Resource Allocation %')
plt.title('Optimized PyTorch - Predicted vs Actual')
plt.tight_layout()
plt.savefig('pytorch_optimized_vs_actual.png')
print("Saved: pytorch_optimized_vs_actual.png")

# ── 13. Save everything ────────────────────────────────────────────
torch.save(model.state_dict(), '5g_model_optimized.pth')
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Saved: 5g_model_optimized.pth")
print("Saved: scaler_X.pkl, scaler_y.pkl, label_encoder.pkl")