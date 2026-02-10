# Deployment Guide — Leaf Disease Classifier

This repository intentionally does not track the dataset. The `models/` directory is tracked so you can include a prebuilt `.keras` model for deployment.

1) Prepare model artifact

- If you already have a trained model locally, upload the single `.keras` file to a stable host (GitHub Release, S3, GCS).
- Alternatively, keep the model in the repo under `models/leaf/current/densenet121/` (committed).

2) Download the model on the target host

Set `MODEL_URL` to point to your hosted model and run the included fetch script:

```bash
# Example (on the VM)
cd /path/to/LeafDiseaseClassification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt   # or requirements-gpu.txt if using GPU

# Download model (replace with your URL)
MODEL_URL="https://my-bucket.s3.amazonaws.com/leaf_densenet121_final.keras" \
  bash scripts/fetch_model.sh
```

The script places the model as `models/leaf/current/densenet121/leaf_densenet121_final.keras`, which the app resolves.

3) Run the app (dev)

```bash
source .venv/bin/activate
python app.py
# or
gunicorn --workers 3 --bind 0.0.0.0:8000 app:app
```

4) Production recommendations

- Run behind `nginx` as a reverse proxy and use `systemd` to manage `gunicorn` (example service below).
- Use HTTPS via certbot.
- Keep `data/` and `outputs/` directories on a separate persistent volume.

Example `systemd` unit (adjust paths and user):

```ini
[Unit]
Description=Leaf Disease Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/LeafDiseaseClassification
Environment="PATH=/home/ubuntu/LeafDiseaseClassification/.venv/bin"
ExecStart=/home/ubuntu/LeafDiseaseClassification/.venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

5) Hosting the model

- GitHub Releases: upload `.keras` as a release asset and use a release URL.
- S3/GCS: stable and performant; use presigned URLs for private artifacts.

6) Keep secrets out of repo

- Do NOT embed private model URLs or cloud keys in the repo. Use environment variables or secret managers.

7) Alternate: include model in repo

- If you prefer a single repo snapshot, commit `models/leaf/current/densenet121/leaf_densenet121_final.keras`. Be mindful of repo size limits (GitHub 100 MB/file hard limit).
