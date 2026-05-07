# Setup And Downloads

This file is the pre-agent checklist. Finish it before starting Phase 00. The plans assume these installs and downloads already exist.

## Grounded Notes

These notes are time-sensitive and were checked for April 16, 2026.

- NVIDIA currently lists `GeForce RTX 5070` under compute capability `12.0`.
- Current bitsandbytes docs list Windows CUDA wheels that include `sm120` targets for modern CUDA toolkits.
- Qwen advises using the latest `transformers`.
- PyTorch public pages are inconsistent right now: the blog shows a `PyTorch 2.11` release dated March 23, 2026, while the public selector page snapshot still renders `Stable (2.7.0)`.

Because of that mismatch, this guide uses the current official Windows CUDA 12.8 pip command. If the live selector later changes, prefer the newer official command and record what you used in your first progress entry.

## 1. Install System Tools

Install these before touching the project environment:

1. Latest stable NVIDIA driver for RTX 5070 from NVIDIA
   Download page: <https://www.nvidia.com/Download/index.aspx>
2. Miniconda for Windows from Anaconda
   Download page: <https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install>
3. Git for Windows
   Download page: <https://git-scm.com/download/win>
4. Optional but recommended: Visual Studio 2022 Build Tools with C++ tools
   Download page: <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
5. Optional but recommended: CMake
   Download page: <https://cmake.org/download/>

Recommended VS Build Tools components:

- MSVC v143 or newer
- Windows 11 SDK
- C++ CMake tools for Windows

Why the optional tools matter:

- bitsandbytes now has Windows wheels, but source-build or dependency recovery is still easier if MSVC and CMake are already present.

## 2. Verify The GPU Driver

Open PowerShell and confirm the driver is visible:

```powershell
nvidia-smi
```

Do not continue until this command works.

## 3. Choose Cache Locations First

Put your model and dataset folders on a fast SSD with comfortable free space.

Recommended minimum free space: 60 GB
Recommended comfortable free space: 80 GB or more

Recommended local folders:

- `C:\AI\models`
- `C:\AI\data`
- `C:\SoftwareProjects\LLMTermProject\external`

You do not need to set cache-related environment variables for this project. Just keep the downloaded files in stable local folders and point the smoke tests or later phase code at those paths.

## 4. Create The Conda Environment

Use a dedicated environment:

```powershell
conda create -n llm-citation python=3.11 -y
conda activate llm-citation
python --version
python -c "import sys; print(sys.executable)"
```

The expected Python family for this project is `3.11.x`.

Before installing packages, keep the environment isolated from user-site Python packages for the current shell:

```powershell
$env:PYTHONNOUSERSITE = '1'
python -c "import site; print(site.getusersitepackages())"
```

Why this matters:

- Windows user-site packages under `C:\Users\<you>\AppData\Roaming\Python\Python311\site-packages` can leak into the conda environment.
- That makes the environment less reproducible and can hide missing dependencies.

## 5. Install PyTorch

Run this inside the `llm-citation` conda environment:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Why this command:

- As of April 16, 2026, the live PyTorch selector page for `Windows + Pip + Python + CUDA 12.8` shows this install pattern.
- CUDA 12.8 is a sensible default for this RTX 5070 setup.

Important:

- Run this install inside the conda environment
- Record the exact command you actually used in your first progress entry
- If the official selector changes later, prefer the updated selector command over this document

Official page:

- <https://pytorch.org/get-started/locally/>

## 6. Install Project Python Packages

After Torch is installed, install the rest:

```powershell
python -m pip install --upgrade pip
python -m pip install -U transformers accelerate bitsandbytes sentence-transformers chromadb rank-bm25 datasets evaluate pandas scikit-learn scipy matplotlib seaborn pillow python-dotenv pytest pypdf pyyaml jsonschema huggingface_hub
```

This package set is intentionally small and direct. Do not add orchestration frameworks before the core pipeline works.

Why `pillow` is included explicitly:

- Some Windows setups silently satisfy it from the user-site directory.
- Installing it directly inside the conda env keeps the project environment self-contained.

## 7. Pre-Download Models

Use the model pages directly instead of Python or CLI download commands.

Recommended local folders:

- `C:\AI\models\QwenQwen2.5-3B-Instruct`
- `C:\AI\models\QwenQwen2.5-7B-Instruct`
- `C:\AI\models\BAAIbge-small-en-v1.5`

Observed on this machine:

- Your downloaded folder names include the publisher name merged into the directory name.
- That is fine. Do not rename them just to match the Hugging Face IDs.
- The smoke script below auto-detects the local folders by structure and name fragments.

### Qwen 2.5 3B Instruct

- Model page: <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>
- Files tab: <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/main>

Download into one folder:

- every `.safetensors` shard listed on the page
- the weights index file if present, such as `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- all tokenizer files shown on the page, especially `tokenizer.json`, `tokenizer_config.json`, and any special-token or vocabulary files that appear

### Qwen 2.5 7B Instruct

- Model page: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- Files tab: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main>

Download into one folder:

- every `.safetensors` shard listed on the page
- the weights index file if present
- `config.json`
- `generation_config.json`
- all tokenizer files shown on the page

### BGE Small Embedding Model

- Model page: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- Files tab: <https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main>

Download into one folder:

- `config.json`
- `modules.json`
- tokenizer files
- all model weight files on the page

Notes:

- The Qwen models are ungated and fit the local-first goal.
- Download both 3B and 7B before the agent starts so the workflow does not stall mid-phase.
- Keep each model in its own folder and do not mix files between models.
- If you download by browser instead of letting Hugging Face cache them automatically, later smoke tests may be easier if you replace remote model IDs with local folder paths.

## 8. Pre-Download Data

### ASQA

Download from the dataset page instead of pulling it with Python.

- Dataset page: <https://huggingface.co/datasets/din0s/asqa>
- Data files page: <https://huggingface.co/datasets/din0s/asqa/tree/main/data>

Download at minimum:

- the `train` parquet file
- the `dev` parquet file
- the dataset card or metadata page for reference

Recommended local folder:

- `C:\AI\data\asqa`

Keep the original file names so Phase 02 can inspect the raw files cleanly.

Observed on this machine:

- The parquet files are currently under `C:\AI\data\asqa\data\`
- The current filenames are:
  - `train-00000-of-00001-87b7d64f7913b544.parquet`
  - `dev-00000-of-00001-58a9a40c6e69f07b.parquet`

If Hugging Face later changes the hashed parquet filenames, download the current `train` and `dev` parquet files from the data page and update the smoke-test paths below to match the files you actually saved.

### ALCE Reference Repo

Download from GitHub directly:

- Repo page: <https://github.com/princeton-nlp/ALCE>
- Direct ZIP download: <https://github.com/princeton-nlp/ALCE/archive/refs/heads/main.zip>

Recommended local folder after extracting:

- `C:\SoftwareProjects\LLMTermProject\external\ALCE_reference`

Important:

- Use ALCE as a reference, not as the project's full runtime stack
- Do not download large retrieval indices or reproduce the old ALCE retrieval pipeline
- Do not depend on ALCE's older environment pins for the main project environment

## 9. Things You Explicitly Do Not Need Up Front

Do not spend time downloading any of the following before the agent starts:

- Full Wikipedia indices
- Large DPR or GTR retrieval assets
- Gated Llama checkpoints
- Real SEC corpora
- Cloud-only evaluation services

The project is scoped to a bounded local corpus and a synthetic finance stress set.

## 10.5 Known Install Warning

You may see a warning like this during `pip install`:

```text
moviepy 2.2.1 requires decorator<6.0,>=4.0.2, which is not installed
```

Interpretation:

- This is usually caused by an unrelated package outside the project dependency set.
- It is not a blocker for this RAG project unless you actually plan to use `moviepy`.

What matters more:

- Torch installed successfully
- The project packages installed successfully
- The smoke tests below pass inside the conda environment

## 10. Smoke Tests Before The Agent Starts

Use the repo script instead of running separate inline snippets.

Script path:

- [scripts/preflight_smoke.py](/C:/SoftwareProjects/LLMTermProject/scripts/preflight_smoke.py)

Run it from the repo root:

```powershell
$env:PYTHONNOUSERSITE = '1'
python scripts\preflight_smoke.py --json-out artifacts\preflight\smoke_test_report.json
```

What the script checks:

- Torch import
- bitsandbytes import
- CUDA availability
- GPU name and compute capability
- Qwen 3B local load and deterministic generation
- BGE embedding model load and encode
- ASQA train and dev parquet load
- ALCE reference folder presence and expected structure

How path detection works:

- It auto-detects the local Qwen 3B folder under `C:\AI\models`
- It auto-detects the local BGE folder under `C:\AI\models`
- It auto-detects the ASQA `train` and `dev` parquet files under `C:\AI\data`
- It checks ALCE at `C:\SoftwareProjects\LLMTermProject\external\ALCE_reference`

Expected result:

- The command exits successfully
- The console prints `[PASS]` for each smoke section
- A JSON report is written to `artifacts\preflight\smoke_test_report.json`

If the quantized Qwen load fails:

```powershell
$env:PYTHONNOUSERSITE = '1'
python scripts\preflight_smoke.py --no-4bit --json-out artifacts\preflight\smoke_test_report.json
```

Use `--no-4bit` only as a fallback diagnostic. The main intended path is the default 4-bit smoke test.

## 10.1 Deeper Verification For Determinism And Attentions

After the basic smoke test passes, run the deeper verification script:

- [scripts/preflight_risk_checks.py](/C:/SoftwareProjects/LLMTermProject/scripts/preflight_risk_checks.py)

Command:

```powershell
$env:PYTHONNOUSERSITE = '1'
python scripts\preflight_risk_checks.py --json-out artifacts\preflight\risk_check_report.json
```

What it verifies:

- repeated deterministic generation produces the same text twice
- `output_attentions=True` returns usable tensors on the eager backend
- a stricter exact-five-words prompt is checked and reported

If the 4-bit path fails and you only want a fallback diagnostic:

```powershell
$env:PYTHONNOUSERSITE = '1'
python scripts\preflight_risk_checks.py --no-4bit --json-out artifacts\preflight\risk_check_report.json
```

If you want the exact-five-words instruction check to be treated as a hard failure:

```powershell
$env:PYTHONNOUSERSITE = '1'
python scripts\preflight_risk_checks.py --strict-instruction --json-out artifacts\preflight\risk_check_report.json
```

## 11. Lock Versions Only After A Successful Smoke Test

After the smoke tests pass, record the environment:

```powershell
New-Item -ItemType Directory -Force artifacts\preflight | Out-Null
python -m pip freeze > artifacts/preflight/requirements-lock.txt
```

Do not create the lock file before the environment is proven to work.

## 12. Fallback Policy

Default path:

1. Windows
2. Conda
3. Latest official Torch Windows CUDA wheel path
4. Qwen 3B bring-up
5. Qwen 7B main experiments

Fallback path:

1. Keep Windows if possible
2. Shorten prompts and use 3B first
3. Resolve quantization or attention-backend issues
4. Move to WSL2 only if Windows quantized inference or attention extraction remains unstable after a documented retry

Do not jump to WSL2 just because it is familiar. Use it only when Windows becomes the actual blocker.

## 13. References

- Miniconda install docs: <https://www.anaconda.com/docs/getting-started/miniconda/install/>
- NVIDIA CUDA GPU list: <https://developer.nvidia.com/cuda/gpus>
- bitsandbytes install docs: <https://huggingface.co/docs/bitsandbytes/installation>
- Qwen2.5-7B-Instruct model card: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- PyTorch blog: <https://pytorch.org/blog/>
- ALCE repo: <https://github.com/princeton-nlp/ALCE>
