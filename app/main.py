from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import soundfile as sf
import os
import re

app = FastAPI()

token = "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"
model_id = "chrisjay/fonxlsr"
processor = Wav2Vec2Processor.from_pretrained(model_id, token=token)
model = Wav2Vec2ForCTC.from_pretrained(model_id, token=token)


def load_audio(file_path: str):
    audio, sampling_rate = torchaudio.load(file_path)
    # Convert to mono channel if not already
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample to 16kHz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio = resampler(audio)
    return audio.squeeze().numpy(), 16000

@app.get("/")
def read_root():
    return {"message": "Welcome to the Speech-to-Text API"}

@app.get("/favicon.ico")
def read_favicon():
    return {"message": "No favicon"}

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    """Endpoint for speech-to-text transcription"""
    # Save uploaded file to a temporary file
    with open("temp_audio", "wb") as f:
        f.write(await file.read())
    
    # Detect file format and load accordingly
    try:
        if file.filename.endswith('.wav'):
            audio_array, sampling_rate = load_audio("temp_audio")
        elif file.filename.endswith('.mp3'):
            audio_array, sampling_rate = librosa.load("temp_audio", sr=16000)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")
    
    # Process audio data
    inputs = processor(audio_array, sampling_rate=16_000, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    
    return {"transcription": transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



