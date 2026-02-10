# System Architecture

Overview
--------
This repo is a leaf disease classification system with a lightweight training pipeline and a Flask web app. The primary components are:

- Dataset storage under `data/leaf_dataset/`
- Training (transfer learning: DenseNet121)
- Inference (leaf prediction)
- Web app (Flask) for uploads and results

High-level pipeline (conceptual)
-------------------------------

```text
Leaf dataset (train/val/test + CSV splits)
         │
         ▼
     [Training]
  - train_leaf_dataset.py
         │
         ▼
  models/leaf/current/densenet121/*.keras
         │
         ▼
      [Inference]
  - predict_leaf.py
         │
         ▼
      [Web App]
  - app.py (Flask) + templates/static
```

Module map (where to look)
--------------------------
- Training: `src/models/train_leaf_dataset.py`
- Inference: `src/inference/predict_leaf.py`
- App & UI: `app.py`, `templates/`, `static/`
- Data: `data/leaf_dataset/` (train/val/test folders and CSV splits)

Data flow
---------
1. **Dataset:** images live under `data/leaf_dataset/` with train/val/test splits.
2. **Training:** `train_leaf_dataset.py` builds a DenseNet121 transfer-learning model.
3. **Artifacts:** models are saved to `models/leaf/current/densenet121/` as timestamped `.keras` files.
4. **Inference:** `predict_leaf.py` loads the latest model and returns class probabilities.
5. **Serving:** `app.py` exposes a Flask UI for browser-based uploads and metrics.

Model details
-------------
- DenseNet121 (transfer learned)
  - Input: (224, 224, 3)
  - Output classes (6): Healthy_Leaf, Healthy_Nut, Mahali_Koleroga, Yellow_Leaf, Ring_Spot, Bud_Rot

Where artifacts are saved
-------------------------
- `models/leaf/current/densenet121/` — saved `.keras` checkpoints
- `outputs/leaf/reports/` — training reports/metrics
- `outputs/predictions/` — inference outputs

Extending the system
--------------------
- To add a new model, create a script under `src/models/` and save artifacts under `models/leaf/current/<model_name>/`.
- To add endpoints, update `app.py` and call `predict_leaf.py` for inference.

Last updated: 2026-02-10
