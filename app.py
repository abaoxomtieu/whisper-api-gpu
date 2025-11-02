from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisper
import uvicorn
import tempfile
import os
import time
from loguru import logger
from typing import List, Optional
from contextlib import asynccontextmanager
from whisper import Whisper

whisper_model: dict[str, Whisper] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    whisper_model["large-v3-turbo"] = whisper.load_model("large-v3-turbo", device="cuda", download_root="/app/models")
    whisper_model["small"] = whisper.load_model(
        "small", device="cuda", download_root="/app/models"
    )
    yield
    for model in whisper_model.values():
        del model

app = FastAPI(title="Whisper API", version="0.1.0", docs_url="/", lifespan=lifespan)

# Load OpenAI Whisper turbo model on GPU (CUDA)
# OpenAI Whisper maps "turbo" to the latest turbo-capable large model


class WordTiming(BaseModel):
    word: str
    start: float
    end: float
    probability: float

class TranslateResponse(BaseModel):
    text: str
    response_time: float
    word_timings: Optional[List[WordTiming]] = None

@app.post("/translate-turbo", response_model=TranslateResponse)
async def translate(file: UploadFile = File(...), include_word_timings: str = Form("false")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Convert string to boolean
    include_word_timings_bool = include_word_timings.lower() in ("true", "1", "yes")

    # Save to a temp file to pass a real path to whisper
    start_time = time.time()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
            
        result = whisper_model["large-v3-turbo"].transcribe(
            tmp_path,
            temperature=0.0,
            prompt="This is a bilingual conversation in English và tiếng Việt:",
            word_timestamps=include_word_timings_bool
        )

        # Get word-level timestamps only if requested
        word_timings = None
        if include_word_timings_bool and "segments" in result:
            word_timings = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timing = WordTiming(
                            word=word_info["word"],
                            start=float(word_info["start"]),
                            end=float(word_info["end"]),
                            probability=float(word_info.get("probability", 1.0))
                        )
                        word_timings.append(word_timing)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Processing time: {processing_time} seconds")
        return JSONResponse(TranslateResponse(
            text=result["text"], 
            response_time=processing_time,
            word_timings=word_timings
        ).model_dump())
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/translate-custom", response_model=TranslateResponse)
async def translate_custom(
    translate_to_english: bool = Form(True),
    temperature: float = Form(0.0), 
    file: UploadFile = File(...), 
    include_word_timings: bool = Form(False)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav"
        ) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
            
        start_time = time.time()
        result = whisper_model["small"].transcribe(
            tmp_path,
            task="translate" if translate_to_english else "transcribe",
            temperature=temperature,
            word_timestamps=include_word_timings
        )

        # Get word-level timestamps only if requested
        word_timings = None
        if include_word_timings and "segments" in result:
            word_timings = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timing = WordTiming(
                            word=word_info["word"],
                            start=float(word_info["start"]),
                            end=float(word_info["end"]),
                            probability=float(word_info.get("probability", 1.0))
                        )
                        word_timings.append(word_timing)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Processing time: {processing_time} seconds")
        return JSONResponse(TranslateResponse(
            text=result["text"], 
            response_time=processing_time,
            word_timings=word_timings
        ).model_dump())
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
