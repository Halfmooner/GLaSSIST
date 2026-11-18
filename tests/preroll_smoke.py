"""Pre-roll smoke test: audio captured just before the wake word fires must be
sent right after the pipeline start request, and the activation chime must be
skipped for wake-word runs when HA_ACK_AT_SPEECH_END is on."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HA_PREROLL_MS"] = "600"
os.environ["HA_ACK_AT_SPEECH_END"] = "true"

import utils  # noqa: E402
import satellite_protocol as sp  # noqa: E402
from aioesphomeapi.api_pb2 import VoiceAssistantAudio, VoiceAssistantRequest  # noqa: E402

played = []
utils.play_feedback_sound = lambda name: played.append(name)
sp.media_ducker.duck = lambda: None

p = sp.VoiceSatelliteProtocol("t", "00:00:00:00:00:00", None, lambda *a: None, lambda: None)
sent = []
p.send_messages = lambda msgs: sent.extend(msgs)

chunk = b"\x01\x00" * 1024  # 64 ms at 16 kHz
for _ in range(30):          # ~1.9 s of idle audio
    p.handle_audio(chunk)
assert not sent, "nothing should be sent while idle"
assert p._preroll_bytes <= 19200 + len(chunk), p._preroll_bytes

p.wakeup()
assert isinstance(sent[0], VoiceAssistantRequest), sent[0]
audio = [m for m in sent[1:] if isinstance(m, VoiceAssistantAudio)]
ms = sum(len(m.data) for m in audio) * 1000 // 32000
assert 500 <= ms <= 700, f"pre-roll {ms} ms"
assert "activation" not in played, played

p.handle_audio(chunk)
assert isinstance(sent[-1], VoiceAssistantAudio)
assert p._preroll_bytes == 0

# Follow-up turn (no wake word): chime plays, no stale pre-roll.
p._pipeline_active = False
p._is_streaming_audio = False
for _ in range(5):
    p.handle_audio(chunk)
n = len(sent)
p._start_pipeline_run()
assert isinstance(sent[n], VoiceAssistantRequest) and len(sent) == n + 1
assert played == ["activation"], played
print(f"preroll ok: {ms} ms sent after wake word")
