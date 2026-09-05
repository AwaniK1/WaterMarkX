Step 1 — Create a Virtual Environment

On Windows:

python -m venv venv

Activate it in PowerShell:

.\venv\Scripts\Activate.ps1

Or in Command Prompt:

venv\Scripts\activate

After activation, the terminal should show something similar to:

(venv) C:\...\kgw-watermarking>

Step 2 — Install Dependencies

Run:

pip install -r requirements.txt

The required libraries are:

torch
transformers==5.15.1
accelerate
sentencepiece

Step 3 - Verify Installation

Check PyTorch:

python -c "import torch; print('PyTorch:', torch.__version__)"

Check Transformers:

python -c "import transformers; print('Transformers:', transformers.__version__)"

Check KGW support:

python -c "from transformers import WatermarkingConfig, WatermarkDetector; print('KGW support: OK')"

The project currently uses:

Transformers 5.15.1

Step 4 - Running the Project

Make sure the virtual environment is activated.

Run:

python kgw_detect.py

The program will ask:

Enter your prompt:

Example:

Explain artificial intelligence in simple words.

Step 5 - First Run

On the first run, Hugging Face will automatically download:

Qwen/Qwen2.5-1.5B-Instruct

Therefore, an internet connection is required during the first run.

The model does not need to be manually downloaded or placed inside the project folder.

After downloading, Transformers can reuse the locally cached model.

What the Program Does

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