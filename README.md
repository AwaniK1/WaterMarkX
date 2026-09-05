# WaterMarkX

A tool for generating and detecting KGW (Kirchenbauer et al.) text watermarks on LLM output, using Qwen2.5-1.5B-Instruct. Includes a Python/FastAPI backend and a React frontend with live generation progress.

---

## Project Structure

    WaterMarkX/
    ├── venv/                  # Python virtual environment (not committed)
    ├── frontend/               # React + Vite + Tailwind UI
    ├── main.py                # FastAPI backend (API used by the frontend)
    ├── kgw_detect.py           # Standalone CLI: generate + detect watermark
    ├── kgw_test.py             # Standalone CLI: generate only
    ├── qwen_test.py            # Standalone CLI: plain Qwen generation, no watermark
    ├── requirements.txt        # Python dependencies
    └── README.md

---

## Backend Setup

### Step 1 — Create a Virtual Environment

On Windows:

```powershell
python -m venv venv
```

Activate it in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Or in Command Prompt:

```text
venv\Scripts\activate
```

After activation, the terminal should show something similar to:

```text
(venv) C:\...\WaterMarkX>
```

### Step 2 — Install Dependencies

Run:

```powershell
pip install -r requirements.txt
```

The required libraries are:

```text
torch
transformers==5.15.1
accelerate
sentencepiece
fastapi
uvicorn[standard]
```

### Step 3 — Verify Installation

Check PyTorch:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
```

Check Transformers:

```powershell
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

Check KGW support:

```powershell
python -c "from transformers import WatermarkingConfig, WatermarkDetector; print('KGW support: OK')"
```

The project currently uses:

```text
Transformers 5.15.1
```

### Step 4 — Running the API Server

Make sure the virtual environment is activated.

Run:

```powershell
uvicorn main:app --reload --port 8000
```

On the first run, Hugging Face will automatically download:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Therefore, an internet connection is required during the first run. The model does not need to be manually downloaded or placed inside the project folder. After downloading, Transformers can reuse the locally cached model.

Once you see:

```text
Model ready on device: ...
Application startup complete.
```

the backend is live at `http://127.0.0.1:8000`. You can test it directly at `http://127.0.0.1:8000/docs`.

### Available Endpoints

| Endpoint           | Method | Purpose                                                  |
|--------------------|--------|-----------------------------------------------------------|
| `/health`          | GET    | Confirms server + model are up                             |
| `/generate`        | POST   | Generates baseline + watermarked text (single response)    |
| `/generate-stream` | POST   | Same as above, but streams live progress + partial text    |
| `/detect`          | POST   | Runs KGW detection on given text, returns z-score etc.      |

---

## Frontend Setup

The frontend lives in `frontend/` — a React app built with Vite and Tailwind CSS.

### Step 1 — Install Dependencies

```powershell
cd frontend
npm install
```

### Step 2 — Run the Dev Server

```powershell
npm run dev
```

Opens at `http://localhost:5173`.

> The frontend calls the backend at `http://127.0.0.1:8000` — make sure the backend is running first.

---

## Running the Full App

Two terminals, both open at once:

**Terminal 1 — backend**

```powershell
cd WaterMarkX
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend**

```powershell
cd WaterMarkX\frontend
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## Standalone CLI Scripts (No Server Needed)

These run independently of the frontend/backend, directly from the terminal — useful for quick testing.

```powershell
python kgw_detect.py
```

The program will ask:

```text
Enter your prompt:
```

Example:

```text
Explain artificial intelligence in simple words.
```

### What the Program Does

The program performs the following steps:

1. Load Qwen
2. Take user prompt
3. Generate normal/original text
4. Configure KGW
5. Generate watermarked text
6. Tokenize both texts
7. Create KGW detector
8. Detect watermark in original text
9. Detect watermark in watermarked text
10. Calculate detection statistics
11. Display results

`kgw_test.py` and `qwen_test.py` are simpler variants (generation only, no CLI input / no watermarking respectively) used during early development.

---

## Notes

- Running on CPU is slow (30–90s per generation). A CUDA-capable GPU will speed this up significantly if available — the backend auto-detects `torch.cuda.is_available()`.
- The KGW watermark uses `greenlist_ratio=0.5`, `bias=2.0`, `seeding_scheme="lefthash"`, `z_threshold=3.0` — see `main.py` to adjust.