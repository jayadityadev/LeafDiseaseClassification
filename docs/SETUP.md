## Setup & Installation — Ubuntu (Linux) and Windows

This guide covers setup for the leaf disease classifier with GPU and CPU workflows.

---

### Quick Commands

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt   # or requirements-cpu.txt
```

Start the app:
```bash
python app.py
```

Optional training:
```bash
python src/models/train_leaf_dataset.py
```

---

### 1) Prerequisites

- **Python:** 3.11 recommended (3.10+ should work).
- **Disk:** ~10 GB free (dataset + models + outputs).
- **Memory:** 8 GB minimum; 16 GB recommended.
- **GPU (optional):** NVIDIA GPU recommended; on smaller GPUs (e.g., GTX 1650) lower batch sizes may be required.

---

### 2) Ubuntu 20.04+ (Native Linux)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git pkg-config
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt   # or requirements-cpu.txt
```

---

### 3) Windows 10 / 11

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt   # or requirements-cpu.txt
```

---

### 4) Optional: Conda Workflow

```bash
conda create -n leafdisease python=3.11
conda activate leafdisease
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-gpu.txt   # or requirements-cpu.txt
```

---

### 5) Verify Installation & Runtime Checks

```bash
python --version
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
```

---

### 6) Troubleshooting

**GPU not detected**
- Verify `nvidia-smi` shows a device and drivers are installed.
- Ensure CUDA/cuDNN versions are compatible with your TensorFlow wheel.

**Out of memory (OOM) on GPU**
- Reduce `BATCH_SIZE` in training scripts (e.g., 32 → 16 → 8).

---

### 7) Environment Variables (example `.env`)
```
FLASK_APP=app.py
FLASK_ENV=production
```
MODEL_TYPE=densenet121
HOST=0.0.0.0
PORT=5000
MAX_CONTENT_LENGTH=16777216
CUDA_VISIBLE_DEVICES=0
```

---

### 10) Success Criteria & Verification

After setup and running preprocessing and validation steps, you should see:

- `python scripts/validate_system.py` → **10/10 tests passed**
- Models saved to `models/current/<model_name>/` (check for final `.keras` file)
- Inference (GPU): ~50–100 ms per image on the hardware used in the verified run
- Prediction outputs saved to `outputs/predictions/`

If you need help diagnosing a specific failure, capture the output of `python scripts/validate_system.py` and `validation_report.txt` and open an issue with those logs.

---

### 11) Advanced / Production Notes (Short)

- **Containerization:** a Dockerfile skeleton exists; for GPU containers see NVIDIA's Docker + CUDA support and the `nvidia-docker` runtime.
- **Model optimization:** consider TF-TRT or ONNX/TensorRT for lower latency in production.
- **Serving:** use Gunicorn + Nginx or a dedicated model server for scaling.

---

Last updated: 2025-11-21
