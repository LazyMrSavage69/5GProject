# 5G Network AI Model

A machine learning pipeline that predicts 5G network resource allocation and signal quality using real network performance data. Built as the AI core of a larger 5G simulation and antenna placement project.

---

## Project Overview

This project trains a neural network to predict 5G network resource allocation based on signal strength, latency, and bandwidth metrics. It is designed to eventually connect to a 3D city map generated from OpenStreetMap data, a Sionna ray tracer for signal propagation simulation, and free5GC for network-layer verification.

```
OSM 3D City Map  →  Sionna Ray Tracing  →  This AI Model  →  Antenna Placement  →  free5GC Verification
```

---

## Current Status

| Component | Status | R2 Score |
|---|---|---|
| Random Forest baseline | Done | 0.8966 |
| PyTorch basic model | Done | 0.9261 |
| PyTorch optimized model | Done | 0.9349 |
| Connect to 3D map | Pending | — |
| Sionna ray tracing | Pending | — |
| Antenna placement optimizer | Pending | — |
| free5GC verification | Pending | — |

---

## Dataset

**Source:** Kaggle — Quality of Service 5G  
**Size:** 400 rows, 8 columns  
**Features used:**

| Column | Description |
|---|---|
| Signal_Strength | Received signal in dBm |
| Latency | Network latency in ms |
| Required_Bandwidth | What the application needs |
| Allocated_Bandwidth | What the network delivers |
| Application_Type | Type of app (video, IoT, etc.) |
| Resource_Allocation | Target variable — efficiency % |

---

## Models

### Random Forest (`5g_model.pkl`)
- 200 trees, max depth 10
- R2: 0.8966 | MAE: 1.12%
- Cross validation average R2: 0.9182

### PyTorch Optimized (`5g_model_optimized.pth`)
- Architecture: 5 → 128 → 256 → 128 → 64 → 1
- BatchNorm + Dropout + AdamW + Early Stopping
- R2: 0.9349 | MAE: 1.23%
- Best performing model

---

## File Structure

```
5GProject/
│
├── train.py                      # Random Forest training
├── train_pytorch.py              # PyTorch optimized training
│
├── Quality of Service 5G.csv     # Dataset
│
├── 5g_model.pkl                  # Saved Random Forest
├── 5g_model_optimized.pth        # Saved PyTorch model
├── scaler_X.pkl                  # Input scaler
├── scaler_y.pkl                  # Output scaler
├── label_encoder.pkl             # App type encoder
│
├── feature_importance.png        # Feature importance chart
├── predicted_vs_actual.png       # Random Forest predictions plot
├── pytorch_predicted_vs_actual.png  # PyTorch predictions plot
└── training_loss_optimized.png   # PyTorch training curve
```

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn
```

### Train Random Forest

```bash
python train.py
```

### Train PyTorch Model

```bash
python train_pytorch.py
```

### Load and Use Saved Model

```python
import torch
import pickle
import numpy as np
from train_pytorch import Network5G

# Load scalers
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Load model
model = Network5G()
model.load_state_dict(torch.load('5g_model_optimized.pth'))
model.eval()

# Predict
# Input: [Signal_Strength, Latency, Required_BW, Allocated_BW, App_encoded]
sample = np.array([[-85, 20, 10, 8, 2]])
sample_scaled = scaler_X.transform(sample)
sample_tensor = torch.FloatTensor(sample_scaled)

with torch.no_grad():
    prediction_scaled = model(sample_tensor).numpy()
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    print(f"Predicted Resource Allocation: {prediction[0][0]:.2f}%")
```

---

## Next Steps

1. Download OpenCelliD or DeepMIMO dataset to continue training with spatial data
2. Connect Sionna ray tracer output to model input pipeline
3. Build antenna placement optimizer on top of model predictions
4. Integrate with free5GC + UERANSIM for network-layer verification
5. Wrap everything in a FastAPI backend with Leaflet.js frontend

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | pandas, numpy, scikit-learn |
| Deep learning | PyTorch |
| 3D city map | osmnx, PyVista, Shapely |
| Signal simulation | Sionna (planned) |
| Network simulation | free5GC + UERANSIM (planned) |
| Backend | FastAPI (planned) |
| Frontend | Leaflet.js (planned) |

---

## Author Notes

- Python 3.10 recommended (3.13 has compatibility issues with ML libraries)
- No GPU required for current model size, but NVIDIA GPU recommended for Sionna phase
- Model performs well at R2 0.93 but is limited by dataset size (400 rows)
- Real performance improvement expected when Sionna spatial data is integrated
