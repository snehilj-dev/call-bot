import asyncio, base64, io, json, logging, audioop, time, os, re, wave
from openai import AsyncOpenAI
import aiohttp
from aiohttp import web, ClientSession

# ----------------------------
# Logging (so Render logs show what's happening)
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ----------------------------
# CONFIG
# ----------------------------
DEEPGRAM_KEY = os.environ.get("DEEPGRAM_KEY", "90332867e648b106891fe713035830598993b4df")
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "").rstrip("/")  # e.g. http://localhost:9000 for Qwen TTS

# ----------------------------
# Helpers: Twilio <-> audio
# ----------------------------
def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s)

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

async def twilio_send_media(twilio_ws, streamSid: str, mulaw_8k_bytes: bytes):
    msg = {
        "event": "media",
        "streamSid": streamSid,
        "media": {"payload": bytes_to_b64(mulaw_8k_bytes)}
    }
    await twilio_ws.send(json.dumps(msg))

async def twilio_clear(twilio_ws, streamSid: str):
    # Interrupt buffered audio (barge-in)
    await twilio_ws.send(json.dumps({"event": "clear", "streamSid": streamSid}))

# ----------------------------
# TTS: Qwen server -> OpenAI fallback -> silence (so call never fails silently)
# Return PCM16 audio bytes + sample_rate (e.g., 24000)
# ----------------------------
def _wav_bytes_to_pcm16_sr(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract PCM16 and sample rate from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    if width != 2 or nch != 1:
        # convert to mono 16-bit if needed (simplified: take first channel and hope)
        raise ValueError(f"Unsupported WAV: nch={nch} sampwidth={width}")
    return (frames, sr)


async def qwen3_tts_pcm16(text: str, instruct: str) -> tuple[bytes, int]:
    """
    Try: (1) TTS_SERVER_URL (Qwen), (2) OpenAI TTS, (3) 0.5s silence.
    Returns (pcm16_bytes, sample_rate).
    """
    # 1) Qwen TTS server (your tts_server.py)
    if TTS_SERVER_URL and text.strip():
        try:
            async with ClientSession() as session:
                async with session.post(
                    f"{TTS_SERVER_URL}/tts",
                    json={"text": text, "instruct": instruct or "Neutral, clear.", "format": "wav"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        wav_bytes = await resp.read()
                        if len(wav_bytes) > 44:
                            pcm, sr = _wav_bytes_to_pcm16_sr(wav_bytes)
                            log.info("TTS: Qwen server OK len=%s sr=%s", len(pcm), sr)
                            return (pcm, sr)
        except Exception as e:
            log.warning("TTS Qwen server failed: %s", e, exc_info=False)

    # 2) OpenAI TTS (PCM 24k)
    if OPENAI_API_KEY and text.strip():
        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            resp = await client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text[:4096],
                response_format="pcm",
                speed=1.0,
            )
            pcm_bytes = resp.content
            if pcm_bytes:
                # OpenAI PCM is 24kHz 16-bit mono
                log.info("TTS: OpenAI OK len=%s", len(pcm_bytes))
                return (pcm_bytes, 24000)
        except Exception as e:
            log.warning("TTS OpenAI failed: %s", e, exc_info=False)

    # 3) Silence so pipeline doesn't crash; user sees in logs that TTS is not configured
    log.warning("TTS: no server and no OpenAI key (or both failed). Playing 0.5s silence. Set TTS_SERVER_URL or OPENAI_API_KEY.")
    silence_8k = 8000 * 1 * 2  # 0.5s mono 16-bit at 8kHz
    return (b"\x00" * silence_8k, 8000)

def pcm16_to_mulaw8k(pcm16: bytes, src_rate: int) -> bytes:
    """
    Convert linear PCM16 at src_rate -> 8k mu-law for Twilio.
    Minimal approach:
      1) resample to 8000 using audioop.ratecv
      2) encode to mu-law using audioop.lin2ulaw
    """
    if src_rate != 8000:
        pcm16, _ = audioop.ratecv(pcm16, 2, 1, src_rate, 8000, None)
    mulaw = audioop.lin2ulaw(pcm16, 2)
    return mulaw

def style_to_instruct(style: dict) -> str:
    # Deterministic mapping: keep it consistent
    tone = style.get("tone", "neutral")
    pace = style.get("pace", "medium")
    breath = style.get("breathiness", "none")
    energy = style.get("energy", "medium")

    parts = []
    if tone == "calm": parts.append("calm, reassuring")
    elif tone == "warm": parts.append("warm, friendly, professional")
    elif tone == "firm": parts.append("firm, clear, commanding")
    else: parts.append("neutral, professional")

    if pace == "slow": parts.append("slightly slower pace")
    elif pace == "fast": parts.append("slightly faster pace")
    else: parts.append("medium pace")

    if breath == "subtle": parts.append("soft breathy warmth")
    elif breath == "moderate": parts.append("breathy, intimate")
    else: parts.append("clean voice")

    if energy == "low": parts.append("low energy, de-escalation")
    elif energy == "high": parts.append("higher energy, confident")
    else: parts.append("controlled energy")

    parts.append("clear diction, subtle pauses")
    return ", ".join(parts)

# ----------------------------
# LLM Planner
# ----------------------------
# Allowed style values for TTS (must match style_to_instruct)
STYLE_TONE = ("calm", "warm", "firm", "neutral")
STYLE_PACE = ("slow", "medium", "fast")
STYLE_BREATH = ("none", "subtle", "moderate")
STYLE_ENERGY = ("low", "medium", "high")

def _normalize_style(s: dict) -> dict:
    """Ensure style dict has only allowed keys and values."""
    tone = s.get("tone", "neutral")
    pace = s.get("pace", "medium")
    breath = s.get("breathiness", "none")
    energy = s.get("energy", "medium")
    gap = int(s.get("gap_ms_after", 200))
    return {
        "tone": tone if tone in STYLE_TONE else "neutral",
        "pace": pace if pace in STYLE_PACE else "medium",
        "breathiness": breath if breath in STYLE_BREATH else "none",
        "energy": energy if energy in STYLE_ENERGY else "medium",
        "gap_ms_after": max(0, min(2000, gap)),
    }

def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response (handles markdown code blocks)."""
    text = text.strip()
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try ```json ... ``` or ``` ... ```
    for pattern in (r"```(?:json)?\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"):
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group(0)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue
    return None

SYSTEM_PROMPT = """You are a voice call agent. You produce SHORT, natural replies for real-time phone calls.

RULES:
- Reply in 1–3 short sentences. Voice-only: no bullets, no "I'll help you with that" fluff unless needed.
- Output valid JSON only, no markdown or explanation.
- Split your reply into 2–5 segments. Each segment is one phrase or short sentence for TTS.
- For each segment set style to control how it's spoken:
  tone: calm | warm | firm | neutral
  pace: slow | medium | fast
  breathiness: none | subtle | moderate
  energy: low | medium | high
  gap_ms_after: 0–2000 (pause in ms after this segment, typically 100–400)
- Use calm/slow/low energy for de-escalation or empathy; warm for greeting; firm for instructions or urgency.
- If risk_flags or negative sentiment: prefer calm, slow, low energy.
- Call state/phase: greeting=opening; troubleshooting=middle; use to keep context."""

async def llm_plan_style(call_state: str, sentiment: str, risk_flags: list[str], call_phase: str, user_text: str) -> dict:
    """
    Return:
      {"reply": str, "segments": [{"text": str, "style": {...}}]}
    """
    if not OPENAI_API_KEY:
        # Fallback: single segment, no API
        reply = "I'm sorry, the assistant isn't configured right now. Please try again later."
        return {
            "reply": reply,
            "segments": [{"text": reply, "style": _normalize_style({})}],
        }

    user_content = (
        f"call_state={call_state!r} call_phase={call_phase!r} sentiment={sentiment!r} "
        f"risk_flags={risk_flags!r}\n\nUser said: {user_text!r}\n\n"
        "Respond with JSON only: {\"reply\": \"full reply text\", \"segments\": [{\"text\": \"...\", \"style\": {\"tone\": \"...\", \"pace\": \"...\", \"breathiness\": \"...\", \"energy\": \"...\", \"gap_ms_after\": number}}]}"
    )

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        raw_text = (resp.choices[0].message.content or "").strip()
    except Exception:
        reply = "I'm having trouble right now. Can you say that again?"
        return {
            "reply": reply,
            "segments": [{"text": reply, "style": _normalize_style({})}],
        }

    out = _extract_json(raw_text)
    if not out or "segments" not in out:
        reply = out.get("reply", raw_text) if isinstance(out, dict) else raw_text
        if not (reply and reply.strip()):
            reply = "I didn't catch that. Could you repeat?"
        return {
            "reply": reply,
            "segments": [{"text": reply.strip(), "style": _normalize_style({})}],
        }

    reply = out.get("reply", "")
    segments = []
    for seg in out.get("segments", []):
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        style = _normalize_style(seg.get("style") or {})
        segments.append({"text": text, "style": style})

    if not segments:
        reply = reply or "Could you say that again?"
        segments = [{"text": reply, "style": _normalize_style({})}]
    else:
        reply = reply or " ".join(s["text"] for s in segments)

    return {"reply": reply, "segments": segments}

# ----------------------------
# Orchestrator per call
# ----------------------------
class CallContext:
    def __init__(self):
        self.call_state = "greeting"
        self.call_phase = "start"
        self.sentiment = "calm"
        self.risk_flags = []
        self.is_speaking = False
        self.last_user_speech_ts = 0.0

async def bridge_twilio_deepgram(twilio_ws):
    ctx = CallContext()
    streamSid = None
    log.info("Call started: bridge_twilio_deepgram entered")

    try:
        dg_headers = {"Authorization": f"Token {DEEPGRAM_KEY}"}
        async with ClientSession() as session:
            async with session.ws_connect(DEEPGRAM_WS_URL, headers=dg_headers) as dg_ws:
                log.info("Deepgram WebSocket connected")

                async def receive_twilio():
                    nonlocal streamSid
                    try:
                        async for raw in twilio_ws:
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError as e:
                                log.warning("Twilio: invalid JSON %s", e)
                                continue
                            ev = msg.get("event")
                            if ev == "start":
                                streamSid = (msg.get("start") or {}).get("streamSid")
                                log.info("Twilio stream started streamSid=%s", streamSid)
                            elif ev == "media":
                                try:
                                    payload = (msg.get("media") or {}).get("payload")
                                    if payload:
                                        audio = b64_to_bytes(payload)
                                        await dg_ws.send_bytes(audio)
                                except Exception as e:
                                    log.warning("Twilio media send failed: %s", e)
                                ctx.last_user_speech_ts = time.time()
                                if ctx.is_speaking and streamSid:
                                    try:
                                        await twilio_clear(twilio_ws, streamSid)
                                    except Exception as e:
                                        log.warning("Twilio clear failed: %s", e)
                                    ctx.is_speaking = False
                            elif ev == "stop":
                                log.info("Twilio stream stop")
                                break
                    except Exception as e:
                        log.exception("receive_twilio error: %s", e)

                async def receive_deepgram_and_respond():
                    nonlocal streamSid
                    try:
                        async for msg in dg_ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                    log.info("Deepgram WS closed or error")
                                    break
                                continue
                            try:
                                dg = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            # Deepgram: support both "channel" and "type":"Results" shapes; use is_final or speech_final
                            channel = dg.get("channel")
                            if not channel:
                                continue
                            alts = (channel.get("alternatives") or [])
                            if not alts:
                                continue
                            alt = alts[0] if isinstance(alts[0], dict) else {}
                            transcript = (alt.get("transcript") or "").strip()
                            if not transcript:
                                continue
                            is_final = dg.get("is_final", False) or dg.get("speech_final", False)
                            if not is_final:
                                continue

                            user_text = transcript
                            log.info("User said: %s", user_text[:200])

                            ctx.call_phase = "middle"
                            if ctx.call_state == "greeting":
                                ctx.call_state = "troubleshooting"

                            try:
                                plan = await llm_plan_style(
                                    call_state=ctx.call_state,
                                    sentiment=ctx.sentiment,
                                    risk_flags=ctx.risk_flags,
                                    call_phase=ctx.call_phase,
                                    user_text=user_text,
                                )
                            except Exception as e:
                                log.exception("LLM failed: %s", e)
                                plan = {
                                    "reply": "I'm having trouble right now. Can you say that again?",
                                    "segments": [{"text": "I'm having trouble. Can you say that again?", "style": {}}],
                                }

                            reply_preview = (plan.get("reply") or "")[:150]
                            log.info("Reply: %s", reply_preview)

                            if not streamSid:
                                continue

                            segments = plan.get("segments") or []
                            for seg in segments:
                                if time.time() - ctx.last_user_speech_ts < 0.4:
                                    break
                                text = (seg.get("text") or "").strip()
                                if not text:
                                    continue
                                try:
                                    instruct = style_to_instruct(seg.get("style") or {})
                                    pcm16, sr = await qwen3_tts_pcm16(text, instruct)
                                    mulaw = pcm16_to_mulaw8k(pcm16, sr)
                                    ctx.is_speaking = True
                                    await twilio_send_media(twilio_ws, streamSid, mulaw)
                                    gap = int((seg.get("style") or {}).get("gap_ms_after", 200))
                                    gap = max(0, min(2000, gap))
                                    await asyncio.sleep(gap / 1000.0)
                                except Exception as e:
                                    log.exception("TTS or send_media failed for segment %r: %s", text[:50], e)
                                finally:
                                    ctx.is_speaking = False
                    except Exception as e:
                        log.exception("receive_deepgram_and_respond error: %s", e)

                await asyncio.gather(receive_twilio(), receive_deepgram_and_respond())
    except Exception as e:
        log.exception("bridge_twilio_deepgram failed: %s", e)
    finally:
        log.info("Call ended: bridge_twilio_deepgram exiting")

# ----------------------------
# HTTP + WebSocket server (health checks use GET/HEAD; websockets library rejects HEAD)
# ----------------------------
class _AiohttpWsAdapter:
    """Wrap aiohttp WebSocket so it matches the interface bridge_twilio_deepgram expects."""
    def __init__(self, ws: web.WebSocketResponse):
        self._ws = ws

    async def send(self, data: str):
        await self._ws.send_str(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self._ws.receive()
        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
            raise StopAsyncIteration
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data
        if msg.type == aiohttp.WSMsgType.BINARY:
            return msg.data.decode("utf-8", errors="replace")
        raise StopAsyncIteration


async def health(_request: web.Request) -> web.Response:
    """Render and load balancers send GET/HEAD to /; respond 200 so health checks pass."""
    return web.Response(text="ok", status=200)


async def ws_twilio(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    log.info("WebSocket /ws/twilio connected")
    try:
        await bridge_twilio_deepgram(_AiohttpWsAdapter(ws))
    except Exception as e:
        log.exception("ws_twilio handler error: %s", e)
    return ws


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"Binding to 0.0.0.0:{port} (PORT={os.environ.get('PORT', '8080')})", flush=True)
    app = web.Application()
    app.router.add_get("/", health)  # aiohttp serves HEAD for GET routes automatically
    app.router.add_get("/ws/twilio", ws_twilio)
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
