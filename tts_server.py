from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Qwen3-TTS Local API")

# Default TTS style: stern warning, commanding, breathy intensity
DEFAULT_INSTRUCT = (
    "Warning anger (stern) + intense breath + commanding. "
    "Stern, angry warning. Slight breathiness, intense pressure in voice. "
    "Strong rise on warning words, heavy low fall at the end. "
    "Medium pace, clear commanding cadence, short pauses between instructions."
)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    speaker: str = "Ryan"          # for CustomVoice
    instruct: str | None = DEFAULT_INSTRUCT  # style instruction (emotion, pace, etc.)
    mode: str = "customvoice"     # "customvoice" or "voicedesign"
    format: str = "wav"           # "wav" or "mp3"

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # CPU-friendly
_model = None

def get_model():
    """Load TTS model on first use (avoids DLL load at import on Windows)."""
    global _model
    if _model is not None:
        return _model
    from qwen_tts import Qwen3TTSModel
    _model = Qwen3TTSModel.from_pretrained(MODEL_ID)
    return _model

@app.get("/")
def root():
    return {"status": "ok", "message": "Qwen3-TTS API. POST /tts with JSON body."}

def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to MP3 using pydub (requires ffmpeg on PATH)."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    out = BytesIO()
    seg.export(out, format="mp3")
    return out.getvalue()

def _wav_arrays_to_bytes(wavs, sample_rate: int) -> bytes:
    """Convert (list of numpy arrays, sample_rate) from generate_custom_voice to WAV bytes."""
    import numpy as np
    from scipy.io import wavfile
    if not wavs or len(wavs) == 0:
        raise ValueError("No audio generated.")
    audio = np.asarray(wavs[0], dtype=np.float64)
    audio = np.squeeze(audio)  # (samples,) or (samples, channels)
    if audio.size == 0:
        raise ValueError("Generated audio is empty.")
    buf = BytesIO()
    # scipy wavfile expects int16 or float32; model returns float in [-1, 1]
    if audio.dtype in (np.float32, np.float64):
        audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(buf, sample_rate, audio)
    return buf.getvalue()

@app.post("/tts")
def tts(req: TTSRequest):
    model = get_model()
    if req.mode.lower() == "customvoice":
        wavs, sample_rate = model.generate_custom_voice(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct
        )
        wav_bytes = _wav_arrays_to_bytes(wavs, sample_rate)
    else:
        raise ValueError("VoiceDesign not enabled in this server example.")

    if not wav_bytes or len(wav_bytes) < 44:  # minimal WAV header
        raise HTTPException(status_code=500, detail="TTS produced no or invalid audio.")

    out_format = (req.format or "wav").strip().lower()
    if out_format == "mp3":
        try:
            audio_bytes = _wav_to_mp3(wav_bytes)
            return Response(content=audio_bytes, media_type="audio/mpeg")
        except Exception:
            # ffmpeg not on PATH (e.g. new install, shell not restarted): return WAV so client still gets audio
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={"X-Audio-Format": "wav", "X-MP3-Unavailable": "ffmpeg not on PATH"},
            )
    return Response(content=wav_bytes, media_type="audio/wav")
