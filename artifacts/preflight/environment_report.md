# Environment Report

## Status

Phase 00 is ready to pass.

- Environment bring-up passed
- 3B risk checks passed
- 7B risk checks passed
- The project is clear to move to Phase 01

## Machine

- OS: `Windows-10-10.0.26200-SP0`
- Python: `3.11.15`
- Python executable: `C:\Users\omehm\anaconda3\envs\llm-citation\python.exe`
- GPU: `NVIDIA GeForce RTX 5070`
- Driver: `591.86`
- VRAM: `12227 MiB`
- Compute capability: `12.0`

## Core Packages

- `torch==2.11.0+cu128`
- `torchvision==0.26.0+cu128`
- `torchaudio==2.11.0+cu128`
- `bitsandbytes==0.49.2`
- `transformers==5.5.4`
- `accelerate==1.13.0`
- `sentence-transformers==5.4.1`
- `datasets==4.8.4`
- `chromadb==1.5.7`

The full lock file is saved at `artifacts/preflight/requirements-lock.txt`.

## Local Assets

- Qwen 3B: `C:\AI\models\QwenQwen2.5-3B-Instruct`
- Qwen 7B: `C:\AI\models\QwenQwen2.5-7B-Instruct`
- BGE small: `C:\AI\models\BAAIbge-small-en-v1.5`
- ASQA train parquet: `C:\AI\data\asqa\data\train-00000-of-00001-87b7d64f7913b544.parquet`
- ASQA dev parquet: `C:\AI\data\asqa\data\dev-00000-of-00001-58a9a40c6e69f07b.parquet`
- ALCE reference: `C:\SoftwareProjects\LLMTermProject\external\ALCE_reference`

## Results

- Torch and bitsandbytes imported successfully in the `llm-citation` environment.
- The 3B model loaded in 4-bit on Windows and generated text successfully.
- The 7B model also passed 4-bit load, deterministic generation, and eager attention checks.
- Both 3B and 7B returned usable `output_attentions=True` tensors for the prompt pass and decode step.
- The attention checks marked both models as usable for token-level gate work.
- ASQA loaded successfully with `4353` train examples and `948` dev examples.

## Advisory Notes

- The exact-five-words instruction-following check failed on both 3B and 7B. This is not a blocker for the project.
- That result supports the planned architecture: prompt instructions alone are not enough for formatting-sensitive behavior, so deterministic validation and verification are still required.
- User-site Python packages exist on this machine. Future work should keep `PYTHONNOUSERSITE=1` in the shell for reproducibility.

## Artifacts

- [smoke_test_report.json](/C:/SoftwareProjects/LLMTermProject/artifacts/preflight/smoke_test_report.json)
- [risk_check_report.json](/C:/SoftwareProjects/LLMTermProject/artifacts/preflight/risk_check_report.json)
- [risk_check_report_7b.json](/C:/SoftwareProjects/LLMTermProject/artifacts/preflight/risk_check_report_7b.json)
- [requirements-lock.txt](/C:/SoftwareProjects/LLMTermProject/artifacts/preflight/requirements-lock.txt)

## Recommendation

Proceed to Phase 01.
