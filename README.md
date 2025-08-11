# Orpheus TTS Server (Canopy Labs)

A minimal, production-oriented FastAPI server for the original Canopy Labs Orpheus TTS model using vLLM.

- Uses OrpheusModel from `orpheus_tts` (original Canopy implementation)
- Applies vLLM stability fixes (EngineDeadError):
  - VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
  - VLLM_WORKER_MULTIPROC_METHOD=spawn
  - Optional: VLLM_ENGINE_ITERATION_TIMEOUT_S, VLLM_GPU_MEMORY_UTILIZATION
- Health, test, and TTS endpoints
- Clear .env for ports and HuggingFace auth

## Requirements

- GPU: NVIDIA A40 (48GB) or similar; Orpheus can use 21–37GB VRAM.
- CUDA 12.x runtime (RunPod CUDA 12 images work well)

## Install (RunPod or local CUDA 12.x)

1) Create and activate Python 3.10/3.11 venv (recommended)

2) Install base deps

```
pip install -r requirements.txt
```

3) Install GPU stack (install torch first, then vllm):

```
# CUDA 12.1 wheels
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install vllm==0.10.0

# Orpheus + SNAC decoder
pip install orpheus-speech==0.1.0 snac==1.2.1
```

4) Hugging Face auth for private/download throttling

```
huggingface-cli login
```

5) Configure environment

Copy `.env.example` to `.env` and adjust as needed.

## Run

```
python app.py
```

Server starts on PORT (default 7007). Endpoints:
- GET /health
- GET /info
- POST /tts  (JSON: { "text": "hello", "voice": "tara", "sample_rate": 24000 }) returns WAV bytes
- GET /      (basic HTML form)

## Notes
- Do not run LLM/STT on the same GPU: Orpheus TTS requires large VRAM.
- If you see EngineDeadError, confirm the env flags are active (see logs at startup).
