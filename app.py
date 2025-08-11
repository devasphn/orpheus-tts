import os
import io
import time
import logging
from typing import Optional

from dotenv import load_dotenv

# Load env early
load_dotenv()

# Apply vLLM stability env before importing orpheus_tts
os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "300")
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import time
import logging
import struct
import asyncio
from typing import Optional
from orpheus_tts import OrpheusModel

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("orpheus-tts-server")

# Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7007"))
MODEL_NAME = os.getenv("ORPHEUS_MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "tara")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))

app = FastAPI(title="Orpheus TTS (Canopy)", version="1.0.0")

# Static/UI
os.makedirs("outputs", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Engine:
    """Singleton-like wrapper to manage model lifecycle."""
    model: Optional[OrpheusModel] = None
    loaded_at: Optional[float] = None
    requests_since_reload: int = 0
    max_requests_before_reload: int = int(os.getenv("MAX_REQUESTS_BEFORE_RELOAD", "10"))

    @classmethod
    def load(cls):
        if cls.model is None:
            logger.info("Initializing OrpheusModel: %s", MODEL_NAME)
            try:
                # Read GPU mem utilization; default to 0.85
                try:
                    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
                except Exception:
                    gpu_mem_util = 0.85

                # Initialize model (vLLM options are controlled by environment; OrpheusModel does not accept these kwargs)
                cls.model = OrpheusModel(
                    model_name=MODEL_NAME
                )
                cls.loaded_at = time.time()
                cls.requests_since_reload = 0
                logger.info("OrpheusModel loaded successfully")
            except Exception as e:
                logger.exception("Failed to initialize OrpheusModel: %s", e)
                raise

    @classmethod
    def maybe_reload(cls):
        # Preventive periodic reload to mitigate vLLM lifecycle issues
        cls.requests_since_reload += 1
        if cls.requests_since_reload >= cls.max_requests_before_reload:
            logger.warning("Reloading Orpheus engine after %d requests", cls.requests_since_reload)
            cls.shutdown()
            cls.load()

    @classmethod
    def shutdown(cls):
        try:
            if cls.model is not None:
                # If model exposes cleanup hooks in future, call here
                pass
        finally:
            cls.model = None


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Orpheus TTS Server on %s:%d", HOST, PORT)
    logger.info("vLLM flags: DISABLE_CUSTOM_ALL_REDUCE=%s, MULTIPROC_METHOD=%s",
                os.getenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE"), os.getenv("VLLM_WORKER_MULTIPROC_METHOD"))
    Engine.load()


@app.on_event("shutdown")
async def on_shutdown():
    Engine.shutdown()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_voice": DEFAULT_VOICE, "sample_rate": SAMPLE_RATE})


@app.get("/health")
async def health():
    ok = Engine.model is not None
    return {"status": "ok" if ok else "down", "model": MODEL_NAME}


@app.get("/info")
async def info():
    return {
        "model": MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "vllm_fixes": {
            "VLLM_DISABLE_CUSTOM_ALL_REDUCE": os.getenv("VLLM_DISABLE_CUSTOM_ALL_REDUCE"),
            "VLLM_WORKER_MULTIPROC_METHOD": os.getenv("VLLM_WORKER_MULTIPROC_METHOD"),
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": os.getenv("VLLM_ENGINE_ITERATION_TIMEOUT_S"),
            "VLLM_GPU_MEMORY_UTILIZATION": os.getenv("VLLM_GPU_MEMORY_UTILIZATION"),
        },
    }


@app.post("/tts")
async def tts(payload: dict):
    if Engine.model is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    prompt = (payload.get("prompt") or payload.get("text") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing text")

    voice = payload.get("voice", DEFAULT_VOICE)
    sample_rate = int(payload.get("sample_rate", SAMPLE_RATE))

    try:
        logger.info(f"Generating TTS for prompt: '{prompt[:50]}...' with voice: {voice}")
        
        def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
            """Create WAV header for streaming"""
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            data_size = 0  # Will be updated later
            
            header = struct.pack(
                '<4sI4s4sIHHIIHH4sI',
                b'RIFF',
                36 + data_size,
                b'WAVE', 
                b'fmt ',
                16,
                1,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                b'data',
                data_size
            )
            return header
        
        def generate_audio_stream():
            """Generator function for streaming audio"""
            try:
                # Send WAV header first
                yield create_wav_header(sample_rate=sample_rate)
                
                # Generate speech chunks
                audio_chunks = Engine.model.generate_speech(
                    prompt=prompt,
                    voice=voice,
                    repetition_penalty=1.1,
                    stop_token_ids=[128258],
                    max_tokens=2000,
                    temperature=0.4,
                    top_p=0.9
                )
                
                chunk_count = 0
                for chunk in audio_chunks:
                    if isinstance(chunk, bytes):
                        yield chunk
                        chunk_count += 1
                    else:
                        yield bytes(chunk)
                        chunk_count += 1
                    
                    # Log progress every 10 chunks
                    if chunk_count % 10 == 0:
                        logger.info(f"Streamed {chunk_count} audio chunks")
                
                logger.info(f"Completed streaming {chunk_count} total audio chunks")
                
            except Exception as e:
                logger.exception(f"Error in audio stream generation: {e}")
                raise
        
        # Return streaming response
        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        # Attempt one soft-reload retry
        Engine.shutdown()
        try:
            Engine.load()
        except Exception as load_err:
            # Likely GPU memory too low; surface clear error
            raise HTTPException(status_code=503, detail=f"Engine reload failed (GPU mem?): {load_err}")
        try:
            logger.info("Retrying TTS generation after engine reload")
            
            def generate_retry_stream():
                try:
                    yield create_wav_header(sample_rate=sample_rate)
                    
                    audio_chunks = Engine.model.generate_speech(
                        prompt=prompt,
                        voice=voice,
                        repetition_penalty=1.1,
                        stop_token_ids=[128258],
                        max_tokens=2000,
                        temperature=0.4,
                        top_p=0.9
                    )
                    
                    chunk_count = 0
                    for chunk in audio_chunks:
                        if isinstance(chunk, bytes):
                            yield chunk
                            chunk_count += 1
                        else:
                            yield bytes(chunk)
                            chunk_count += 1
                    
                    logger.info(f"Retry completed: {chunk_count} chunks streamed")
                    
                except Exception as e:
                    logger.exception(f"Error in retry stream: {e}")
                    raise
            
            return StreamingResponse(
                generate_retry_stream(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav",
                    "Cache-Control": "no-cache"
                }
            )
        except Exception as e2:
            logger.exception("Retry failed: %s", e2)
            raise HTTPException(status_code=500, detail=f"TTS failed: {e2}")

    Engine.maybe_reload()

    raise HTTPException(status_code=500, detail="Unexpected code path")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT)
