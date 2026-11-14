"""
Application utility functions module.
"""
import os
import logging
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import platform_utils

load_dotenv(platform_utils.get_env_file_path())

def safe_print(text):
    """
    Print text safely, handling Unicode encoding issues on Windows.
    
    Args:
        text: The text to print (string)
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Handle encoding issues by encoding to UTF-8 with error replacement
        safe_text = str(text).encode('utf-8', errors='replace').decode('utf-8')
        print(safe_text)

def setup_logger():
    """Return the shared logger; repeated calls do not recreate files/handlers."""
    from logging_config import configure_logging
    configure_logging(
        get_env_bool('DEBUG', False),
        platform_utils.get_config_dir() / 'logs',
    )
    return logging.getLogger('haassist')


def get_env_bool(key, default=False):
    """Get environment variable as boolean with safe parsing."""
    value = get_env(key, str(default).lower())
    
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'y', 't')
    else:
        return bool(value)

def get_env(key, default=None, as_type=str):
    """Get environment variable and optionally convert to specified type."""
    value = _read_from_env_file(key)
    
    if value is None:
        value = os.getenv(key, default)
    
    if value is None:
        return None
    
    if as_type == bool:
        # Safe boolean parsing
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'y', 't')
        else:
            return bool(value)
    
    try:
        return as_type(value)
    except (ValueError, TypeError):
        logger.warning(f"Cannot convert '{value}' to type {as_type.__name__} for key {key}")
        return default

def _read_from_env_file(key):
    """Read value directly from .env file."""
    possible_paths = [
        str(platform_utils.get_env_file_path()),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),
        '.env'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                env_key, env_value = parts
                                if env_key.strip() == key:
                                    return env_value.strip()
            except Exception as e:
                logger.warning(f"Error reading .env file: {e}")
    
    return None

def get_output_device_index():
    """Return configured output device index or None for automatic selection."""
    output_device = get_env("HA_OUTPUT_DEVICE_INDEX", -1, int)
    if output_device is None or output_device < 0:
        return None
    return output_device

def get_output_sample_rate():
    """Return configured output sample rate or None for automatic selection."""
    output_rate = get_env("HA_OUTPUT_SAMPLE_RATE", -1, int)
    if output_rate is None or output_rate <= 0:
        return None
    return output_rate

def normalize_audio_device_name(value):
    """Repair PortAudio UTF-8 names misdecoded by Windows legacy code pages.

    Strict round trips leave correctly decoded Unicode intact. Normalize only
    after repair: decomposing the mojibake first destroys reversible bytes.
    """
    import unicodedata
    if isinstance(value, bytes):
        try:
            value = value.decode('utf-8')
        except UnicodeDecodeError:
            value = value.decode('mbcs' if os.name == 'nt' else 'cp1252', errors='replace')
    value = str(value).replace('\x00', '').strip()
    for encoding in ('cp1250', 'cp1252', 'latin1'):
        try:
            repaired = value.encode(encoding).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired != value:
            value = repaired
            break
    return unicodedata.normalize('NFC', value)


def get_available_output_devices():
    """Return a list of available output devices."""
    devices = []
    try:
        for index, device_info in enumerate(sd.query_devices()):
            if device_info.get('max_output_channels', 0) > 0:
                device_name = device_info.get('name', f"Output {index}")
                device_name = normalize_audio_device_name(device_name)
                if not device_name:
                    device_name = f"Output {index}"
                
                devices.append({
                    'index': index,
                    'name': device_name,
                    'channels': device_info.get('max_output_channels', 0),
                    'sample_rate': device_info.get(
                        'default_samplerate',
                        device_info.get('defaultSampleRate', 'unknown')
                    )
                })
    except Exception as e:
        logger.error(f"Failed to query output devices: {e}")
    
    return devices

def _resample_audio(audio_data, original_rate, target_rate):
    """Resample audio data to the target rate."""
    if original_rate == target_rate or target_rate is None:
        return audio_data

    try:
        from scipy.signal import resample_poly
    except Exception as e:
        logger.warning(f"Resampling unavailable: {e}")
        return audio_data

    try:
        import math
        original_rate = int(original_rate)
        target_rate = int(target_rate)
        if original_rate <= 0 or target_rate <= 0:
            return audio_data

        divisor = math.gcd(original_rate, target_rate)
        up = target_rate // divisor
        down = original_rate // divisor

        data = audio_data
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype.kind in ("i", "u"):
            data = data.astype(np.float32)
        else:
            data = data.astype(np.float32)

        return resample_poly(data, up, down, axis=0)
    except Exception as e:
        logger.warning(f"Failed to resample audio: {e}")
        return audio_data

def get_timestamp():
    """Return current timestamp in milliseconds."""
    return int(time.time() * 1000)

def get_datetime_string():
    """Return formatted date and time string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _looks_like_mp3(content):
    """True for an ID3 tag or an MPEG audio frame sync at the start."""
    return content[:3] == b'ID3' or (len(content) > 1 and content[0] == 0xFF
                                     and content[1] & 0xE0 == 0xE0)

def _decode_audio_bytes(content, logger):
    """
    Decode downloaded audio into (float32 data, samplerate).
    MP3 goes through miniaudio first: it decodes frame by frame to the end,
    while libsndfile trusts the length estimate of VBR files without a Xing
    header and drops the tail (issue #55). Windows has no ffmpeg for audioread.
    """
    if _looks_like_mp3(content):
        try:
            import miniaudio
            decoded = miniaudio.mp3_read_f32(content)
            data = np.asarray(decoded.samples, dtype=np.float32)
            if decoded.nchannels > 1:
                data = data.reshape((-1, decoded.nchannels))
            logger.info(f"Decoded with miniaudio: {len(data)} frames, "
                        f"{decoded.sample_rate}Hz, {decoded.nchannels}ch")
            return data, decoded.sample_rate
        except ImportError:
            pass  # only installed on Windows (requirements.txt)
        except Exception as e:
            logger.warning(f"miniaudio failed, trying audioread: {e!r}")

    import tempfile
    tmp_path = None
    try:
        import audioread
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(content)

        with audioread.audio_open(tmp_path) as f:
            samplerate = f.samplerate
            channels = f.channels
            audio_bytes = b''.join(f)

        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if channels == 2:
            data = data.reshape((-1, 2))
        logger.info(f"Decoded with audioread: {len(data)} samples, {samplerate}Hz, {channels}ch")
        return data, samplerate
    except Exception as e:
        # NoBackendError (no ffmpeg/gstreamer) has an empty message, hence repr
        logger.warning(f"audioread failed, falling back to soundfile: {e!r}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return sf.read(io.BytesIO(content))

def explicit_stream_disabled():
    """HA_STREAM_TTS=false turns progressive playback off."""
    return not get_env_bool("HA_STREAM_TTS", True)


class _HttpStreamSource:
    """Seekable view over a chunked HTTP body; keeps what it has read so the
    decoder can seek back while probing, and blocks for data it has not got."""

    def __init__(self, response, chunk_size=4096):
        self._it = response.iter_content(chunk_size=chunk_size)
        self._buf = bytearray()
        self._pos = 0
        self._eof = False

    def _fill(self, upto):
        while len(self._buf) < upto and not self._eof:
            try:
                chunk = next(self._it)
            except StopIteration:
                self._eof = True
                break
            if chunk:
                self._buf.extend(chunk)

    def read(self, num_bytes):
        self._fill(self._pos + num_bytes)
        data = bytes(self._buf[self._pos:self._pos + num_bytes])
        self._pos += len(data)
        return data

    def seek(self, offset, origin):
        import miniaudio
        new = offset if origin == miniaudio.SeekOrigin.START else self._pos + offset
        if new < 0:
            return False
        self._fill(new)
        if new > len(self._buf):
            return False
        self._pos = new
        return True


def _try_stream_mp3(full_url, logger, animation_server=None):
    """
    Decode and play an MP3 while it is still downloading, so speech starts
    with the first frames instead of after the whole file has arrived.
    Returns True/False when it handled playback, or None to let the caller
    fall back to download-then-play (no miniaudio, not MP3, early failure).
    """
    try:
        import miniaudio
    except ImportError:
        return None

    try:
        response = requests.get(full_url, timeout=10, stream=True)
    except Exception as e:
        logger.warning(f"Streaming request failed, falling back: {e!r}")
        return None
    if response.status_code != 200:
        response.close()
        return None
    ctype = (response.headers.get("Content-Type") or "").lower()
    if "mpeg" not in ctype and "mp3" not in ctype and not full_url.lower().split("?")[0].endswith(".mp3"):
        response.close()
        return None

    class _Source(miniaudio.StreamableSource):
        def __init__(self, inner):
            self.inner = inner

        def read(self, num_bytes):
            return self.inner.read(num_bytes)

        def seek(self, offset, origin):
            return self.inner.seek(offset, origin)

    try:
        out_rate = get_output_sample_rate()
        if not out_rate:
            try:
                out_rate = int(sd.query_devices(get_output_device_index(), "output")["default_samplerate"])
            except Exception:
                out_rate = 48000
        source = _Source(_HttpStreamSource(response))
        frames = max(256, out_rate // 50)  # ~20 ms per chunk
        gen = miniaudio.stream_any(
            source,
            source_format=miniaudio.FileFormat.MP3,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=out_rate,
            frames_to_read=frames,
        )
        # Pre-buffer ~150 ms before opening the device to avoid an initial underrun.
        pending = []
        buffered = 0
        for chunk in gen:
            arr = np.frombuffer(chunk, dtype=np.float32)
            if arr.size:
                pending.append(arr)
                buffered += arr.size
            if buffered >= out_rate * 0.15:
                break
        if not pending:
            response.close()
            return None
    except Exception as e:
        logger.warning(f"Streaming decode failed, falling back: {e!r}")
        try:
            response.close()
        except Exception:
            pass
        return None

    logger.info(f"Streaming playback ({out_rate}Hz)")
    try:
        device = get_output_device_index()
        with sd.OutputStream(device=device, samplerate=out_rate, channels=1,
                             dtype="float32") as output:
            def _emit(arr):
                output.write(arr.reshape(-1, 1))
                if animation_server:
                    _send_fft_to_animation((arr * 32767).astype(np.int16), animation_server)

            for arr in pending:
                _emit(arr)
            for chunk in gen:
                arr = np.frombuffer(chunk, dtype=np.float32)
                if arr.size:
                    _emit(arr)
        logger.info("Streaming playback completed")
        return True
    except Exception as e:
        logger.exception(f"Streaming playback error: {e}")
        return False
    finally:
        try:
            response.close()
        except Exception:
            pass


def play_audio_from_url(url, host, animation_server=None, done_callback=None,
                        output_device_index=None):
    """
    Play audio from given URL using sounddevice and soundfile.
    Optionally send FFT data to animation server during playback.
    done_callback is called when playback finishes (used by ESPHome satellite mode).
    """
    logger = setup_logger()
    
    if not url:
        logger.error("No audio file URL provided")
        return False
    
    try:
        import re as _re
        _is_local = os.path.isabs(url) or _re.match(r'^[A-Za-z]:[/\\]', url)

        if _is_local:
            local_path = os.path.normpath(url)
            logger.info(f"Playing local audio file: {local_path}")
            data, samplerate = sf.read(local_path)
        else:
            if url.startswith('/'):
                if host.startswith(('localhost', '127.0.0.1', '192.168.', '10.', '172.')):
                    protocol = "http"
                else:
                    protocol = "https"
                full_url = f"{protocol}://{host}{url}"
            else:
                full_url = url

            if not explicit_stream_disabled() and output_device_index is None:
                streamed = _try_stream_mp3(full_url, logger, animation_server)
                if streamed is not None:
                    if done_callback:
                        try:
                            done_callback()
                        except Exception as e:
                            logger.error(f"done_callback error: {e}")
                    return streamed

            logger.info(f"Downloading audio from: {full_url}")

            response = requests.get(full_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Audio download error: {response.status_code}")
                return False

            audio_buffer = io.BytesIO(response.content)

            logger.info("Reading audio file...")
            data, samplerate = _decode_audio_bytes(response.content, logger)
        explicit_output = output_device_index is not None
        if not explicit_output:
            output_device_index = get_output_device_index()
        output_sample_rate = get_output_sample_rate()
        if output_device_index is not None:
            logger.debug(f"Using output device index: {output_device_index}")
        if output_sample_rate is not None and output_sample_rate != samplerate:
            logger.info(f"Resampling audio from {samplerate}Hz to {output_sample_rate}Hz")
            data = _resample_audio(data, samplerate, output_sample_rate)
            samplerate = output_sample_rate
        if explicit_output:
            # A dedicated stream avoids sounddevice.play's process-global stream,
            # which can interrupt another room or a feedback sound.
            channels = 1 if data.ndim == 1 else data.shape[1]
            with sd.OutputStream(device=output_device_index, samplerate=samplerate,
                                 channels=channels, dtype='float32') as output:
                output.write(data.astype(np.float32))
            result = True
        elif animation_server:
            logger.info(f"Playing with FFT analysis (samplerate: {samplerate})...")
            result = _play_with_fft_analysis(
                data,
                samplerate,
                animation_server,
                output_device_index
            )
        else:
            logger.info(f"Standard playback (samplerate: {samplerate})...")
            if output_device_index is not None:
                sd.play(data, samplerate, device=output_device_index)
            else:
                sd.play(data, samplerate)
            sd.wait()
            logger.info("Audio playback completed")
            result = True

        if done_callback:
            try:
                done_callback()
            except Exception as e:
                logger.error(f"done_callback error: {e}")
        return result

    except Exception as e:
        logger.exception(f"Audio playback error: {str(e)}")
        if done_callback:
            try:
                done_callback()
            except Exception:
                pass
        return False

def _play_with_fft_analysis(audio_data, samplerate, animation_server, output_device_index=None):
    """
    Play audio and simultaneously send FFT data to animation server.
    Uses simple sd.play() + sd.wait() for reliable playback.
    """
    logger = setup_logger()

    try:
        chunk_size = 1024  # FFT chunk size

        # Convert stereo to mono for FFT analysis (keep original for playback)
        if len(audio_data.shape) > 1:
            fft_data = np.mean(audio_data, axis=1)
        else:
            fft_data = audio_data

        # Convert FFT data to int16 for animation
        if fft_data.dtype != np.int16:
            if np.abs(fft_data).max() <= 1.0:
                fft_data = (fft_data * 32767).astype(np.int16)
            else:
                fft_data = fft_data.astype(np.int16)

        total_samples = len(fft_data)
        duration = total_samples / samplerate
        logger.info(f"Starting playback with FFT analysis... Duration: {duration:.2f}s")

        # Start FFT analysis in background thread
        stop_fft = threading.Event()

        def fft_thread_func():
            """Send FFT data to animation server during playback."""
            samples_sent = 0
            start_time = time.time()

            while not stop_fft.is_set() and samples_sent < total_samples:
                elapsed = time.time() - start_time
                target_sample = int(elapsed * samplerate)

                if target_sample > samples_sent and samples_sent < total_samples:
                    chunk_end = min(samples_sent + chunk_size, total_samples)
                    chunk_data = fft_data[samples_sent:chunk_end]

                    if len(chunk_data) > 0:
                        _send_fft_to_animation(chunk_data, animation_server)

                    samples_sent = chunk_end

                time.sleep(0.02)  # ~50 FPS for smooth animation

        fft_thread = threading.Thread(target=fft_thread_func, daemon=True)
        fft_thread.start()

        # Play audio using simple sd.play() + sd.wait()
        # This is the most reliable method
        play_kwargs = {}
        if output_device_index is not None:
            play_kwargs['device'] = output_device_index

        sd.play(audio_data, samplerate, **play_kwargs)
        sd.wait()  # Block until playback is complete

        # Stop FFT thread
        stop_fft.set()
        fft_thread.join(timeout=1.0)

        logger.info("FFT analysis playback completed")
        return True

    except Exception as e:
        logger.exception(f"FFT analysis playback error: {str(e)}")
        return False

def _send_fft_to_animation(audio_chunk, animation_server):
    """
    Perform FFT analysis and send data to animation server.
    Called in separate thread.
    """
    try:
        if len(audio_chunk) == 0:
            return
            
        if audio_chunk.dtype != np.int16:
            audio_chunk = audio_chunk.astype(np.int16)
        audio_chunk = (audio_chunk * 0.6).astype(np.int16)
        audio_bytes = audio_chunk.tobytes()
        
        animation_server.send_audio_data(audio_bytes, 16000)
        
    except Exception as e:
        pass

def validate_audio_format(sample_rate, channels=1):
    """Validate audio parameters."""
    valid_rates = [8000, 16000, 22050, 44100, 48000]
    if sample_rate not in valid_rates:
        logger = setup_logger()
        logger.warning(f"Unusual sample rate: {sample_rate}Hz")
    
    if channels not in [1, 2]:
        logger = setup_logger()
        logger.warning(f"Unusual channel count: {channels}")
    
    return True

DEFAULT_SOUND_FILES = {
    'activation': 'activation.wav',
    'deactivation': 'deactivation.wav',
    'processing': 'processing.wav',
}

def get_sound_dir():
    """Return path to the bundled 'sound' folder."""
    return os.path.join(os.path.dirname(__file__), 'sound')

def get_sound_file_path(sound_name):
    """
    Resolve the configured file path for a feedback sound role.

    Args:
        sound_name: Sound role ('activation', 'deactivation' or 'processing')
    """
    default = DEFAULT_SOUND_FILES.get(sound_name, f"{sound_name}.wav")
    filename = get_env(f'HA_SOUND_{sound_name.upper()}', default)
    if not filename:
        filename = default
    user_dir = platform_utils.get_user_sound_dir()
    if user_dir is not None:
        user_path = os.path.join(str(user_dir), filename)
        if os.path.exists(user_path):
            return user_path
    return os.path.join(get_sound_dir(), filename)

def play_feedback_sound(sound_name):
    """
    Play feedback sound from 'sound' folder.

    Args:
        sound_name: Sound role ('activation' or 'deactivation'), file
                    configurable via HA_SOUND_ACTIVATION/HA_SOUND_DEACTIVATION
    """
    sound_enabled = get_env('HA_SOUND_FEEDBACK', 'true')
    if sound_enabled.lower() not in ('true', '1', 'yes', 'y', 't'):
        return False

    logger = setup_logger()
    output_device_index = get_output_device_index()
    output_sample_rate = get_output_sample_rate()

    try:
        sound_file = get_sound_file_path(sound_name)

        if not os.path.exists(sound_file):
            logger.warning(f"Sound file not found: {sound_file}")
            return False
        
        def play_thread():
            try:
                data, samplerate = sf.read(sound_file)
                if output_sample_rate is not None and output_sample_rate != samplerate:
                    logger.info(
                        f"Resampling feedback sound from {samplerate}Hz to {output_sample_rate}Hz"
                    )
                    data = _resample_audio(data, samplerate, output_sample_rate)
                    samplerate = output_sample_rate
                if output_device_index is not None:
                    sd.play(data, samplerate, device=output_device_index)
                else:
                    sd.play(data, samplerate)
                
            except Exception as e:
                logger.error(f"Error playing sound {sound_name}: {e}")
        
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
        
        logger.debug(f"Playing sound: {sound_name}")
        return True
        
    except Exception as e:
        logger.error(f"Feedback sound playback error {sound_name}: {e}")
        return False

class ProcessingSoundLoop:
    """
    Loops the configured processing sound while waiting for a response.

    Enabled via HA_PROCESSING_SOUND, file configurable via HA_SOUND_PROCESSING.
    start() is a no-op when disabled or the file is missing; stop() is idempotent.
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        logger = setup_logger()

        enabled = get_env('HA_PROCESSING_SOUND', 'false')
        if enabled.lower() not in ('true', '1', 'yes', 'y', 't'):
            return False

        sound_file = get_sound_file_path('processing')
        if not os.path.exists(sound_file):
            logger.warning(f"Processing sound file not found: {sound_file}")
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(sound_file,), daemon=True
            )
            self._thread.start()

        logger.debug("Processing sound loop started")
        return True

    def _loop(self, sound_file):
        logger = setup_logger()
        try:
            data, samplerate = sf.read(sound_file)
            output_sample_rate = get_output_sample_rate()
            output_device_index = get_output_device_index()
            if output_sample_rate is not None and output_sample_rate != samplerate:
                data = _resample_audio(data, samplerate, output_sample_rate)
                samplerate = output_sample_rate
            duration = len(data) / float(samplerate)

            while not self._stop_event.is_set():
                if output_device_index is not None:
                    sd.play(data, samplerate, device=output_device_index)
                else:
                    sd.play(data, samplerate)
                if self._stop_event.wait(duration):
                    sd.stop()
                    break
        except Exception as e:
            logger.error(f"Processing sound loop error: {e}")

    def stop(self):
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

processing_sound_loop = ProcessingSoundLoop()

def format_duration(seconds):
    """Format duration in seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

logger = setup_logger()
