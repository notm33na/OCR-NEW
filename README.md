# Bilingual OCR API (FastAPI)

Production FastAPI service for **Urdu** (YOLOv8 + UTRNet) and **English** (TrOCR) OCR. Script detection for `auto` mode. CPU-only safe; cross-platform; UTF-8 safe.

---

## Endpoints

| Method | Path    | Description |
|--------|--------|-------------|
| GET    | `/health` | Readiness check. Returns `{"status": "ok"}`. |
| POST   | `/ocr`    | Run OCR on uploaded image (multipart/form-data). |

### POST /ocr

- **Request:** `multipart/form-data`
  - `image`: file (jpg/png)
  - `language`: `"urdu"` \| `"english"` \| `"auto"` (default: `auto`)
- **Response:** JSON
  - `language_detected`: `"urdu"` \| `"english"`
  - `text`: full recognized text (combined)
  - `regions`: list of `{ "id": int, "language": "urdu"|"english", "text": "..." }`

Example (curl):

```bash
curl -X POST "http://localhost:8000/ocr" \
  -F "image=@sample_inputs/urdu_printed_1.jpg" \
  -F "language=urdu"
```

---

## Local run (Windows / Linux)

1. **Python 3.9+**, virtualenv recommended.

2. **Install dependencies**
   - Windows (CPU):  
     `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`  
     then `pip install -r requirements.txt`
   - Linux (e.g. Railway):  
     `pip install -r requirements.txt` (use default PyTorch or CPU index if needed).

3. **Models and data**
   - UTRNet: clone `UTRNet-High-Resolution-Urdu-Text-Recognition` and place `saved_models/UTRNet-Large/best_norm_ED.pth`.
   - YOLOv8 text detection: for region detection (same as Railway), run  
     `python scripts/download_urdudoc_model.py`  
     to download `urdu-text-detection/yolov8m_UrduDoc.pt`. Otherwise the pipeline uses `yolov8n.pt` + OpenCV fallback.
   - Tesseract: required for script detection (`auto`); install and add to PATH.

4. **Start the app**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   Or with reload: `uvicorn app:app --reload --port 8000`

5. **Health check:** `GET http://localhost:8000/health`  
   **Docs:** `http://localhost:8000/docs`

---

## Deployment (Railway)

1. **Connect** the repo to Railway; root = project root.

2. **Build**
   - Railway runs `pip install -r requirements.txt` by default.
   - On Linux, if you use Windows-only PyTorch pins (`torch==2.10.0+cpu`), replace with generic `torch` in `requirements.txt` or set a **Nixpacks/Railway** build command that installs PyTorch CPU first, then the rest.

3. **Start command** (Railway uses `Procfile`):
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
   Railway sets `PORT`; the app binds to `0.0.0.0` and `$PORT`.

4. **Environment**
   - No required env vars for basic run.
   - Optional: `TMPDIR` or system temp is used for uploads (cross-platform).

5. **Assets on deploy**
   - Include UTRNet repo and `saved_models/UTRNet-Large/best_norm_ED.pth` in the repo or attach a volume so the app can load the model at startup.
   - YOLOv8 weights are downloaded on first use if missing.

6. **.railwayignore**
   - Used to skip `.git`, `__pycache__`, venvs, and optionally large `inputs/` or `outputs/` so deploys stay small.

---

## Engineering notes

- **CPU-only:** No GPU required; all inference runs on CPU.
- **Paths:** Uses `pathlib` and `tempfile.gettempdir()`; no hardcoded `C:\`.
- **UTF-8:** Stdout/stderr reconfigured on Windows; responses are JSON (UTF-8).
- **Models:** Loaded once at startup (UTRNet, TrOCR); YOLOv8 loads on first use.
- **Async:** OCR runs in a thread pool so the event loop is not blocked.
