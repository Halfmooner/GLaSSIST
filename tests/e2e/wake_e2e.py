"""End-to-end wake word test: real openWakeWord detector + real ESPHome satellite
server + a fake Home Assistant (aioesphomeapi client), fed with synthesized
speech in real time through a fake microphone.

Usage: python wake_e2e.py <speech.wav> <protocol_dir> <models> [rounds]
  protocol_dir: directory whose satellite_protocol.py is used (lets us compare
                the old and new protocol code under identical conditions).
Prints a JSON summary line starting with RESULT.
"""
import asyncio
import json
import os
import sys
import threading
import time
import wave

import numpy as np

wav_path, proto_dir, models = sys.argv[1], sys.argv[2], sys.argv[3]
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 12
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.update({
    "HA_WAKE_WORD_ENABLED": "true",
    "HA_WAKE_WORD_MODELS": models,
    "HA_WAKE_WORD_THRESHOLD": os.environ.get("E2E_THRESHOLD", "0.5"),
    "HA_WAKE_WORD_VAD_THRESHOLD": "0.3",
    "HA_SOUND_FEEDBACK": "true",
    "HA_PROCESSING_SOUND": "false",
})
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(proto_dir))  # protocol under test wins

import satellite_protocol  # noqa: E402
import wake_word_detector  # noqa: E402
print("protocol module:", satellite_protocol.__file__, flush=True)

RATE = 16000

# ---------------------------------------------------------------- timeline
with wave.open(wav_path) as w:
    assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
    speech = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
rng = np.random.default_rng(1)
def noise(sec):
    return (rng.normal(0, 60, int(sec * RATE))).astype(np.int16)
parts, utter_at = [noise(2.0)], []
pos = len(parts[0])
for _ in range(rounds):
    utter_at.append(pos / RATE)
    parts += [speech, noise(5.0)]
    pos += len(speech) + int(5.0 * RATE)
timeline = np.concatenate(parts)
DURATION = len(timeline) / RATE
T0 = None  # set when the clock starts


class FakeStream:
    """Real-time paced reader over the shared timeline (like a microphone)."""
    def __init__(self):
        self.pos = 0
    def read(self, n, exception_on_overflow=False):
        while T0 is None:
            time.sleep(0.01)
        due = T0 + (self.pos + n) / RATE
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.5:  # fell behind: drop audio like a real overflowing device
            self.pos = int((time.monotonic() - T0) * RATE) - n
        chunk = timeline[self.pos:self.pos + n]
        self.pos += n
        if len(chunk) < n:
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), np.int16)])
        return chunk.tobytes()
    def stop_stream(self): pass
    def close(self): pass


class FakePyAudio:
    def get_default_input_device_info(self): return {"index": 0, "name": "fake-mic"}
    def get_device_count(self): return 1
    def get_device_info_by_index(self, i): return {"index": 0, "name": "fake-mic", "maxInputChannels": 1}
    def open(self, **kw): return FakeStream()
    def terminate(self): pass

wake_word_detector.pyaudio.PyAudio = FakePyAudio

# ---------------------------------------------------------------- satellite
detections, runs, audio_per_run = [], [], []
tts_done = []

def on_tts_url(url, done_callback=None):
    def _play():
        time.sleep(1.0)  # pretend to play the answer
        tts_done.append(time.monotonic() - T0)
        if done_callback:
            done_callback()
    threading.Thread(target=_play, daemon=True).start()

server = satellite_protocol.SatelliteServer(
    device_name="e2e", animation_server=None, on_tts_url=on_tts_url,
    on_tts_finished=lambda: None, port=16053)

def on_wake(model_name, confidence):
    print(f"wake {model_name} {confidence:.3f} at {time.monotonic() - T0:.1f}s", flush=True)
    detections.append((round(time.monotonic() - T0, 2), model_name, round(float(confidence), 3)))
    server.wakeup()

detector = wake_word_detector.WakeWordDetector(callback=on_wake)
assert detector.model is not None, "wake word model failed to load"
loaded = sorted(detector.model.models.keys())
print("loaded models:", loaded, flush=True)


def mic_thread():
    s = FakeStream()
    chunk = int(RATE * 0.03)
    while time.monotonic() - (T0 or time.monotonic()) < DURATION + 1:
        server.handle_audio(s.read(chunk))


async def fake_home_assistant():
    from aioesphomeapi import APIClient, VoiceAssistantEventType as E
    cli = APIClient("127.0.0.1", 16053, None)
    await cli.connect(login=True)
    loop = asyncio.get_running_loop()
    current = {"bytes": 0}

    async def run(conv_id):
        t = time.monotonic() - T0
        runs.append(round(t, 2))
        current["bytes"] = 0
        send = cli.send_voice_assistant_event
        send(E.VOICE_ASSISTANT_RUN_START, {})
        send(E.VOICE_ASSISTANT_STT_START, {})
        send(E.VOICE_ASSISTANT_STT_VAD_START, {})
        await asyncio.sleep(2.5)  # user speaks the command
        send(E.VOICE_ASSISTANT_STT_VAD_END, {})
        audio_per_run.append(current["bytes"] * 1000 // (RATE * 2))
        send(E.VOICE_ASSISTANT_STT_END, {"text": "turn on the light"})
        send(E.VOICE_ASSISTANT_INTENT_START, {})
        await asyncio.sleep(0.3)
        send(E.VOICE_ASSISTANT_INTENT_END, {"conversation_id": "x", "continue_conversation": "0"})
        send(E.VOICE_ASSISTANT_TTS_START, {"text": "ok"})
        send(E.VOICE_ASSISTANT_TTS_END, {"url": "http://127.0.0.1:1/x.mp3"})
        await asyncio.sleep(0.1)
        send(E.VOICE_ASSISTANT_RUN_END, {})

    async def handle_start(conversation_id, flags, audio_settings, wake_word_phrase=None, *a):
        loop.create_task(run(conversation_id))
        return 0

    async def handle_stop(*a):
        pass

    async def handle_audio(data):
        current["bytes"] += len(data)

    cli.subscribe_voice_assistant(handle_start=handle_start, handle_stop=handle_stop,
                                  handle_audio=handle_audio)
    return cli


async def main():
    global T0
    global STAGE
    await server.start()
    STAGE = "connect"
    cli = await asyncio.wait_for(fake_home_assistant(), 20)
    STAGE = "running"
    print("fake HA connected", flush=True)
    await asyncio.sleep(1.0)
    T0 = time.monotonic()
    detector.start_detection()
    threading.Thread(target=mic_thread, daemon=True).start()
    await asyncio.sleep(DURATION + 6)
    detector.stop_detection()
    STAGE = "shutdown"
    try:
        await asyncio.wait_for(cli.disconnect(), 5)
    except Exception as e:
        print("disconnect:", e, flush=True)


def _watchdog():
    print("RESULT " + json.dumps({"error": "timeout", "stage": STAGE,
          "detections": detections, "runs": runs}), flush=True)
    os._exit(2)

STAGE = "start"
threading.Timer(DURATION + 90, _watchdog).start()
asyncio.run(main())

# match each utterance with a run that started within 2.5 s after it began
hit = 0
for u in utter_at:
    if any(u <= r <= u + 2.5 for r in runs):
        hit += 1
result = {
    "protocol": proto_dir, "models": models, "loaded": loaded,
    "utterances": len(utter_at), "detections": len(detections),
    "runs": len(runs), "utterances_with_run": hit,
    "audio_ms_at_vad_end": audio_per_run, "tts_done": len(tts_done),
    "detection_log": detections,
}
print("RESULT " + json.dumps(result), flush=True)
os._exit(0)
