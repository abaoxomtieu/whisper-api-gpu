from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisper
import uvicorn
import tempfile
import os
import time
from loguru import logger
from whisper.tokenizer import get_tokenizer
from whisper.timing import find_alignment
from typing import List, Optional

app = FastAPI(title="Whisper API", version="0.1.0",docs_url="/")

# Load OpenAI Whisper turbo model on GPU (CUDA)
# OpenAI Whisper maps "turbo" to the latest turbo-capable large model
_MODEL_NAME = "large-v3-turbo"
model = whisper.load_model("large-v3-turbo", device="cuda", download_root="/app/models")


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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            options = whisper.DecodingOptions(
                temperature=0.0,
                prompt="This is a bilingual conversation in English và tiếng Việt:"

            )
            result = whisper.decode(model, mel, options)

            # Get word-level timestamps only if requested
            word_timings = None
            if include_word_timings_bool:
                tokenizer = get_tokenizer(multilingual=True)
                text_tokens = tokenizer.encode(result.text)

                alignments = find_alignment(
                    model=model,
                    tokenizer=tokenizer,
                    text_tokens=text_tokens,
                    mel=mel,
                    num_frames=mel.shape[-1],
                )

                # Convert alignments to WordTiming objects
                word_timings = []
                for alignment in alignments:
                    word_timing = WordTiming(
                        word=alignment.word,
                        start=float(alignment.start),
                        end=float(alignment.end),
                        probability=float(alignment.probability)
                    )
                    word_timings.append(word_timing)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Processing time: {processing_time} seconds")
        return JSONResponse(TranslateResponse(
            text=result.text, 
            response_time=processing_time,
            word_timings=word_timings
        ).model_dump())
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

model_custom = whisper.load_model("small", device="cuda", download_root="/app/models")


@app.post("/translate-custom", response_model=TranslateResponse)
async def translate_custom(
    translate_to_english: bool = Form(True),
    temperature: float = Form(0.0), 
    file: UploadFile = File(...), 
    include_word_timings: bool = Form(False)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav"
        ) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
            start_time = time.time()
            audio = whisper.load_audio(tmp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model_custom.device)
            options = whisper.DecodingOptions(
                task="translate" if translate_to_english else "transcribe",
                temperature=temperature,
            )
            result = whisper.decode(model_custom, mel, options)

            # Get word-level timestamps only if requested
            word_timings = None
            if include_word_timings:
                tokenizer = get_tokenizer(multilingual=True)
                text_tokens = tokenizer.encode(result.text)

                alignments = find_alignment(
                    model=model_custom,
                    tokenizer=tokenizer,
                    text_tokens=text_tokens,
                    mel=mel,
                    num_frames=mel.shape[-1],
                )

                # Convert alignments to WordTiming objects
                word_timings = []
                for alignment in alignments:
                    word_timing = WordTiming(
                        word=alignment.word,
                        start=float(alignment.start),
                        end=float(alignment.end),
                        probability=float(alignment.probability)
                    )
                    word_timings.append(word_timing)

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Processing time: {processing_time} seconds")
        return JSONResponse(TranslateResponse(
            text=result.text, 
            response_time=processing_time,
            word_timings=word_timings
        ).model_dump())
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
