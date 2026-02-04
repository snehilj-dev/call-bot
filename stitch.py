from pydub import AudioSegment

gap_ms = 250
silence = AudioSegment.silent(duration=gap_ms)

parts = ["s1.wav", "s2.wav", "s3.wav", "s4.wav", "s5.wav"]

final = AudioSegment.empty()
for p in parts:
    final += AudioSegment.from_wav(p) + silence

final.export("final.mp3", format="mp3", bitrate="64k")
print("Saved final.mp3")
