"""OpenAI-compatible audio transcription server (CPU).

Endpoints:
    POST /v1/audio/transcriptions
        Standard OpenAI endpoint. Extends the response with a `translation`
        field when the request includes a `translate_to` form field (ISO-639
        code such as "pt", "es", "fr"). Clients that ignore unknown fields
        still work as if talking to plain OpenAI.
    POST /v1/audio/translations
        Standard OpenAI endpoint: transcribe any language directly into
        English using Whisper's `translate` task.
    GET  /v1/models
    GET  /health

Environment variables:
    WHISPER_MODEL          default "small"  e.g. tiny, base, small, medium, large-v3
    WHISPER_THREADS       default 4        number of CPU threads
    WHISPER_MODELS_DIR    default ~/.whisper  directory for GGML models
    TRANSLATE_TO           default unset    default target language for the
                                            `translation` field when the client
                                            does not send `translate_to`.
    HOST, PORT             default 0.0.0.0:8000
    LOG_LEVEL              default INFO

Run:
    python main.py
or:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from transcriber import Transcriber
from translator import Translator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("legenda.server")

AVAILABLE_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
}

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
N_THREADS = int(os.getenv("WHISPER_THREADS", "4"))
MODELS_DIR = os.getenv("WHISPER_MODELS_DIR") or None
DEFAULT_TRANSLATE_TO = os.getenv("TRANSLATE_TO") or None

transcribers: Dict[str, Transcriber] = {}
translator = Translator()


def get_transcriber(model_size: str) -> Transcriber:
    if model_size not in transcribers:
        log.info("loading whisper model %s (threads=%d)", model_size, N_THREADS)
        transcribers[model_size] = Transcriber(
            model_size=model_size,
            n_threads=N_THREADS,
            models_dir=MODELS_DIR,
        )
    return transcribers[model_size]


if DEFAULT_TRANSLATE_TO:
    try:
        translator.ensure("en", DEFAULT_TRANSLATE_TO)
        log.info("preinstalled en->%s translation package", DEFAULT_TRANSLATE_TO)
    except Exception as e:
        log.warning("could not preinstall en->%s: %s", DEFAULT_TRANSLATE_TO, e)

log.info("ready")

app = FastAPI(title="Legenda STT", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "whisper.cpp",
            }
            for model_id in AVAILABLE_MODELS
        ],
    }


def _save_upload(upload: UploadFile, data: bytes) -> str:
    import subprocess

    suffix = ""
    if upload.filename:
        _, ext = os.path.splitext(upload.filename)
        suffix = ext or ""
    if not suffix:
        suffix = ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="legenda-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if suffix != ".wav":
            wav_path = tempfile.mktemp(suffix=".wav", prefix="legenda-")
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True,
                check=True,
            )
            os.unlink(path)
            return wav_path
    except Exception:
        os.unlink(path)
        raise
    return path


def _verbose_payload(result, language_hint):
    return {
        "task": "transcribe",
        "language": result.language or language_hint,
        "duration": result.duration,
        "text": result.text,
        "segments": [
            {
                "id": i,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for i, s in enumerate(result.segments)
        ],
    }


def _transcription_payload(result, response_format: str, language_hint):
    if response_format == "verbose_json":
        return _verbose_payload(result, language_hint)
    return {"text": result.text}


async def _run_transcribe(
    upload: UploadFile,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    task: str,
):
    transc = get_transcriber(model)
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio file")
    path = _save_upload(upload, data)
    try:
        return await asyncio.to_thread(
            transc.transcribe,
            path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            task=task,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """Standard OpenAI transcription endpoint. Pure `{"text": ...}` response."""
    result = await _run_transcribe(
        file, model, language, prompt, temperature, task="transcribe"
    )
    if response_format == "text":
        return PlainTextResponse(result.text)
    return JSONResponse(_transcription_payload(result, response_format, language))


@app.post("/v1/audio/translations")
async def create_openai_translation(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """Standard OpenAI translations endpoint (any language -> English, via Whisper)."""
    result = await _run_transcribe(
        file, model, None, prompt, temperature, task="translate"
    )
    if response_format == "text":
        return PlainTextResponse(result.text)
    return JSONResponse({"text": result.text})


@app.post("/v1/audio/dual")
async def create_dual(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    target_language: Optional[str] = Form(None),
):
    """Dual output: raw transcription + translation, both as OpenAI-shaped objects.

    Runs Whisper once and the translator once on the same audio chunk and
    returns:

        {
            "transcription": { ...OpenAI transcription response... },
            "translation":   { "text": "..." }
        }

    Each inner object is a valid OpenAI response; clients can pass them
    straight to code that expects an OpenAI transcription/translation result.
    """
    target = (target_language or DEFAULT_TRANSLATE_TO or "").strip() or None
    result = await _run_transcribe(
        file, model, language, prompt, temperature, task="transcribe"
    )

    transcription_body = _transcription_payload(result, response_format, language)

    translation_text = ""
    if target and result.text:
        src = result.language or language or "en"
        if src == target:
            translation_text = result.text
        else:
            try:
                translation_text = await asyncio.to_thread(
                    translator.translate, result.text, src, target
                )
            except Exception as e:
                log.warning("translation failed (%s->%s): %s", src, target, e)
                translation_text = ""

    translation_body = {"text": translation_text}

    if response_format == "text":
        return PlainTextResponse(f"{result.text}\n---\n{translation_text}")

    return JSONResponse(
        {
            "transcription": transcription_body,
            "translation": translation_body,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
