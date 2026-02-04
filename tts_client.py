"""Call the local TTS API and save audio to a file. Usage:
  python tts_client.py "Your text here"
  python tts_client.py "Your text here" --format mp3
  python tts_client.py "Your text here" --out my_audio.wav
  python tts_client.py "Your text here" --instruct "Calm, gentle narration. Slow pace."
  python tts_client.py "Your text here" --instruct "Calm narration" --format mp3
"""
import argparse
import sys

def main():
    p = argparse.ArgumentParser(description="TTS client: save API response to WAV or MP3")
    p.add_argument("text", help="Text to speak")
    p.add_argument("--url", default="http://127.0.0.1:9000", help="TTS API base URL")
    p.add_argument("--format", "-f", choices=("wav", "mp3"), default="wav", help="Output format (default: wav)")
    p.add_argument("--out", "-o", default=None, help="Output filename (default: out.wav or out.mp3)")
    p.add_argument("--instruct", "-i", default=None, help="Style instruction (emotion, pace, etc.). Omit to use server default.")
    p.add_argument("--speaker", "-s", default="Ryan", help="Speaker name (default: Ryan)")
    p.add_argument("--timeout", "-t", type=int, default=600, help="Request timeout in seconds (default: 600). Use more for long text or first run.")
    args = p.parse_args()

    try:
        import urllib.request
        import json
    except ImportError:
        print("Need urllib.request and json (standard library).", file=sys.stderr)
        sys.exit(1)

    out_file = args.out or f"out.{args.format}"
    payload = {
        "text": args.text,
        "language": "English",
        "speaker": args.speaker,
        "format": args.format,
    }
    if args.instruct is not None:
        payload["instruct"] = args.instruct
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/tts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        print(f"API error {e.code}: {e.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        err = str(e).lower()
        if "timed out" in err or "timeout" in err:
            print(f"Request timed out after {args.timeout}s. Long text or first run can take several minutes. Try --timeout 900.", file=sys.stderr)
        else:
            print(f"Request failed (is the server running?): {e}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("Empty response from API.", file=sys.stderr)
        sys.exit(1)

    with open(out_file, "wb") as f:
        f.write(data)
    print(f"Saved {len(data)} bytes to {out_file}")


if __name__ == "__main__":
    main()
