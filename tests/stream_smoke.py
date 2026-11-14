"""Smoke test for progressive MP3 playback (run on Windows CI with miniaudio).

Serves a 3 s tone slowly (~1.5 s to download), replaces the sound device
with a recorder and checks that audio starts before the download finishes
and that the whole clip is played.
"""
import http.server
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd

import utils

DATA = open(os.path.join(os.path.dirname(__file__), "data", "tone_3s.mp3"), "rb").read()
DOWNLOAD_SECONDS = 1.5
state = {"done_at": None}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.end_headers()
        step = max(1, len(DATA) // 30)
        for i in range(0, len(DATA), step):
            self.wfile.write(DATA[i:i + step])
            self.wfile.flush()
            time.sleep(DOWNLOAD_SECONDS / 30)
        state["done_at"] = time.time()

    def log_message(self, *a):
        pass


class FakeStream:
    def __init__(self, samplerate, **kw):
        self.rate = samplerate
        self.samples = 0
        self.first = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, arr):
        if self.first is None:
            self.first = time.time()
        self.samples += len(arr)


created = []


def fake_output_stream(*a, **kw):
    s = FakeStream(**kw)
    created.append(s)
    return s


sd.OutputStream = fake_output_stream
os.environ["HA_OUTPUT_SAMPLE_RATE"] = "48000"

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/api/tts_proxy/x.mp3"

calls = []
t0 = time.time()
ok = utils.play_audio_from_url(url, "", None, done_callback=lambda: calls.append(1))
assert ok is True, ok
assert calls == [1], "done_callback not called"
assert created, "streaming path was not used"
s = created[0]
played = s.samples / s.rate
print(f"first audio after {s.first - t0:.2f}s, download done after {state['done_at'] - t0:.2f}s, "
      f"played {played:.2f}s of audio")
assert s.first < state["done_at"] - 0.5, "playback did not start before download finished"
assert 2.8 < played < 3.3, f"unexpected played length {played}"
print("STREAM SMOKE TEST PASSED")
