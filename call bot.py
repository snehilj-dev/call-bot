import asyncio, base64, json, audioop, time, os, re
from websockets import connect
from openai import AsyncOpenAI
import aiohttp
from aiohttp import web

# ----------------------------
# CONFIG
# ----------------------------
#"YOUR_DEEPGRAM_KEY"
DEEPGRAM_KEY = "90332867e648b106891fe713035830598993b4df"
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")  # or "gpt-4o" for stronger reasoning

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
# TTS: placeholder
# Return PCM16 audio bytes + sample_rate (e.g., 24000)
# ----------------------------
async def qwen3_tts_pcm16(text: str, instruct: str) -> tuple[bytes, int]:
    """
    Replace this with your real Qwen3-TTS call.
    Must return linear PCM 16-bit little-endian bytes and its sample rate.
    """
    raise NotImplementedError

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

    # Connect to Deepgram STT
    dg_headers = {"Authorization": f"Token {DEEPGRAM_KEY}"}
    async with connect(DEEPGRAM_WS_URL, extra_headers=dg_headers) as dg_ws:

        async def receive_twilio():
            nonlocal streamSid
            async for raw in twilio_ws:
                msg = json.loads(raw)
                ev = msg.get("event")

                if ev == "start":
                    streamSid = msg["start"]["streamSid"]

                elif ev == "media":
                    # Inbound caller audio (mulaw 8k) -> Deepgram
                    payload = msg["media"]["payload"]
                    audio = b64_to_bytes(payload)
                    await dg_ws.send(audio)

                    # Barge-in detection heuristic:
                    # if user audio is arriving while we are speaking, interrupt
                    ctx.last_user_speech_ts = time.time()
                    if ctx.is_speaking and streamSid:
                        await twilio_clear(twilio_ws, streamSid)
                        ctx.is_speaking = False  # stop current agent output

                elif ev == "stop":
                    break

        async def receive_deepgram_and_respond():
            nonlocal streamSid
            buffer_text = ""

            async for raw in dg_ws:
                dg = json.loads(raw)

                # Deepgram message shapes vary; look for final transcript fields in your chosen API mode.
                # Here we assume dg["channel"]["alternatives"][0]["transcript"] and dg["is_final"]
                if "channel" not in dg: 
                    continue

                alt = dg["channel"]["alternatives"][0]
                transcript = alt.get("transcript", "").strip()
                if not transcript:
                    continue

                is_final = dg.get("is_final", False)
                if not is_final:
                    continue

                user_text = transcript  # final utterance

                # Update state machine (simple example)
                ctx.call_phase = "middle"
                if ctx.call_state == "greeting":
                    ctx.call_state = "troubleshooting"

                # Ask LLM for plan
                plan = await llm_plan_style(
                    call_state=ctx.call_state,
                    sentiment=ctx.sentiment,
                    risk_flags=ctx.risk_flags,
                    call_phase=ctx.call_phase,
                    user_text=user_text
                )

                # Speak segments with dynamic styles
                if not streamSid:
                    continue

                for seg in plan["segments"]:
                    # If user barged in recently, stop speaking
                    if time.time() - ctx.last_user_speech_ts < 0.4:
                        break

                    instruct = style_to_instruct(seg["style"])
                    pcm16, sr = await qwen3_tts_pcm16(seg["text"], instruct)
                    mulaw = pcm16_to_mulaw8k(pcm16, sr)

                    ctx.is_speaking = True
                    await twilio_send_media(twilio_ws, streamSid, mulaw)

                    # Gap control (dynamic pause)
                    gap = int(seg["style"].get("gap_ms_after", 200))
                    await asyncio.sleep(gap / 1000.0)

                ctx.is_speaking = False

        await asyncio.gather(receive_twilio(), receive_deepgram_and_respond())

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
    await bridge_twilio_deepgram(_AiohttpWsAdapter(ws))
    return ws


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"Binding to 0.0.0.0:{port} (PORT={os.environ.get('PORT', '8080')})", flush=True)
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_head("/", health)
    app.router.add_get("/ws/twilio", ws_twilio)
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
