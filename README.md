# Leaf Disease Classification

Deep-learning system to classify six arecanut leaf conditions:

- Healthy_Leaf
- Healthy_Nut
- Mahali_Koleroga
- Yellow_Leaf
- Ring_Spot
- Bud_Rot

The dataset is already prepared under `data/leaf_dataset/` with train/val/test splits and CSVs (`train_split.csv`, `test_split.csv`).

---

## Quick Start

### 1) Create and activate a virtual environment

**Windows / PowerShell**
```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

**Linux / macOS (bash)**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2) Install dependencies (choose GPU or CPU)

```bash
# GPU (recommended when you have a compatible NVIDIA GPU)
pip install -r requirements-gpu.txt

# or CPU-only (slower, but no GPU required)
# pip install -r requirements-cpu.txt
```

### 3) Run the web app

```bash
python app.py
```

Open <http://localhost:5000> and upload a leaf image.

### 4) Train a new model (optional)

```bash
python src/models/train_leaf_dataset.py
```

Trained models are saved under `models/leaf/current/densenet121/` as timestamped `.keras` files.

---

## Project Structure (Key Files)

```text
LeafDiseaseClassification/
├── app.py                          # Flask web app for uploads
├── src/
│   ├── models/
│   │   └── train_leaf_dataset.py   # Leaf training pipeline (DenseNet121)
│   └── inference/
│       └── predict_leaf.py         # Leaf inference (text-only results)
├── data/
│   └── leaf_dataset/               # Train/val/test folders + CSV splits
├── models/
│   └── leaf/current/densenet121/   # Saved leaf models (.keras)
├── outputs/                        # Reports, predictions, logs
├── docs/                           # Setup + architecture docs
├── templates/                      # HTML templates for the web UI
└── static/                         # CSS/JS assets for the web UI
```

---

## Tech Stack

- Python 3.11
- TensorFlow/Keras
- OpenCV, Pillow
- NumPy, scikit-learn
- Flask

---

## Documentation

- `docs/ARCHITECTURE.md` — system architecture and data flow
- `docs/SETUP.md` — environment setup and troubleshooting
- `docs/DOCKER.md` — CPU-only Docker usage

---

## Troubleshooting

**GPU not detected**
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

**Module import errors**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

*Last Updated: February 10, 2026*
## 📝 License

This is a research/educational project.

## 👤 Author

**Jayaditya Dev**
- Email: jayadityadev261204@gmail.com
- GitHub: [@jayadityadev](https://github.com/jayadityadev)

---

*Last Updated: November 21, 2025*