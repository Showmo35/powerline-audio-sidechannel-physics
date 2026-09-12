#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Powerline Leakage Monitor
# Author: xingya
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import os
    import sys

    # The xcb Qt platform plugin transitively loads Snap's libpthread, which is
    # ABI-incompatible with the system glibc and causes a symbol lookup crash.
    # Re-exec with the real system libpthread preloaded to shadow Snap's copy.
    _PTHREAD = '/lib/x86_64-linux-gnu/libpthread.so.0'
    if (sys.platform.startswith('linux')
            and os.path.exists(_PTHREAD)
            and _PTHREAD not in os.environ.get('LD_PRELOAD', '')):
        os.environ['LD_PRELOAD'] = _PTHREAD
        os.execv(sys.executable, [sys.executable] + sys.argv)

    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
from gnuradio import filter as gr_filter
import sip
from gnuradio import blocks
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
import os
import glob
import subprocess
import threading
import time
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd

from gnuradio import qtgui

# ── constants ─────────────────────────────────────────────────────────────────

CAPTURE_PATH   = '/home/user/Documents/Capture_Analysis/capture.bin'
AUDIO_EXTS     = ('*.wav', '*.mp3', '*.flac', '*.ogg', '*.aac', '*.m4a')
# PulseAudio / PipeWire sink to use for audio output (monitor HDMI)
AUDIO_SINK     = 'alsa_output.pci-0000_00_1f.3.hdmi-stereo'
# Seconds to wait after ffplay exits before reading the end-of-audio byte
# offset.  Gives the GNURadio pipeline (complex_to_float → file_sink) time to
# drain its internal buffers so the marker lands on the actual audio boundary.
PIPELINE_DRAIN_S = 0.1
# Extra recording time kept after audio ends so the powerline tail (delayed by
# the physical lag of the amplifier/PSU, up to ~700 ms) is fully captured.
# align_lag() trims this off the front after detecting the actual lag, so the
# saved capture.bin ends up exactly audio_duration long.
LAG_TAIL_S = 0.800

# ── helpers ───────────────────────────────────────────────────────────────────

def find_audio_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer the canonical reference clip if it's present — the directory also
    # contains reconstruction *outputs* (am_reconstruct*.wav) that sort first
    # alphabetically but must NOT be played back into the soundbar as the source.
    preferred = os.path.join(script_dir, 'hello5.wav')
    if os.path.exists(preferred):
        return preferred
    for pattern in AUDIO_EXTS:
        matches = sorted(glob.glob(os.path.join(script_dir, pattern)))
        if matches:
            return matches[0]
    return None

def _fsize(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def get_audio_duration(audio_path):
    """Return exact duration in seconds via ffprobe, or None on failure."""
    if not audio_path:
        return None
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'csv=p=0', audio_path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None

def trim_capture(capture_path, start_bytes, end_bytes, samp_rate, audio_duration_s=None):
    """In-place trim capture_path so sample 0 == audio t=0 and length == audio duration.

    Step 1 — window shift: discard the pre-audio preamble by copying
              [start_bytes : end_bytes] to the front of the file.
    Step 2 — exact truncation: if audio_duration_s is provided, truncate to
              exactly round(audio_duration_s * samp_rate) samples, removing
              any pipeline-drain overshoot at the tail.

    Must be called AFTER self.wait() so the file sink has flushed all buffers.
    """
    if start_bytes is None:
        print("[capture] No alignment markers — skipping trim")
        return

    # Align to float32 (4-byte) boundary
    start = (start_bytes // 4) * 4
    end   = (end_bytes   // 4) * 4 if end_bytes is not None else None

    try:
        total = os.path.getsize(capture_path)
    except OSError:
        print("[capture] capture.bin missing — cannot trim")
        return

    if end is None or end > total:
        end = total
    if start >= end:
        print(f"[capture] Degenerate window (start={start} >= end={end}) — skipping")
        return

    n_bytes    = end - start
    duration_s = n_bytes / 4 / samp_rate

    print(f"[capture] Step 1 — window shift")
    print(f"          {total/1e6:.3f} MB total  →  keeping {n_bytes/1e6:.3f} MB ({duration_s:.4f}s)")
    print(f"          preamble removed : {start/4/samp_rate:.4f}s")
    print(f"          tail removed     : {(total-end)/4/samp_rate:.4f}s")

    # In-place shift: read from [start:end], write back from byte 0.
    # Safe because read_pos always leads write_pos by `start` bytes (no overlap).
    CHUNK = 4 * 1024 * 1024   # 4 MB per chunk
    try:
        with open(capture_path, 'r+b') as f:
            read_pos  = start
            write_pos = 0
            remaining = n_bytes
            while remaining > 0:
                f.seek(read_pos)
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                f.seek(write_pos)
                f.write(data)
                n = len(data)
                read_pos  += n
                write_pos += n
                remaining -= n
            f.truncate(write_pos)
    except Exception as exc:
        print(f"[capture] Trim error: {exc}")
        return

    # ── Step 2: clip any hardware overshoot beyond the capture window ────────
    # The end marker was set LAG_TAIL_S after audio ended, so the window
    # already contains the full powerline tail. Just remove stray samples that
    # arrived after capture_end_bytes was read (race with the file sink).
    if audio_duration_s is not None:
        target_samples = int(round((audio_duration_s + LAG_TAIL_S) * samp_rate))
        target_bytes   = target_samples * 4
        current_bytes  = os.path.getsize(capture_path)
        overshoot_s    = (current_bytes - target_bytes) / 4 / samp_rate

        print(f"[capture] Step 2 — clip to audio ({audio_duration_s:.6f}s) + {LAG_TAIL_S*1000:.0f} ms tail")
        if target_bytes <= current_bytes:
            os.truncate(capture_path, target_bytes)
            print(f"          clipped {overshoot_s*1000:.1f} ms overshoot")
        else:
            print(f"          capture is {-overshoot_s*1000:.1f} ms short of target — no padding")

    final = os.path.getsize(capture_path)
    final_s = (final // 4) / samp_rate
    print(f"[capture] Done — capture.bin = {final/1e6:.3f} MB  ({final_s:.6f}s)")

def align_lag(capture_path, audio_path, samp_rate, aud_sr=22050, frame_s=0.05):
    """Detect and remove the powerline lag from capture.bin in-place.

    After trim_capture() the capture is time-aligned to audio start but the
    powerline signal is still physically delayed (DSP / power-supply latency).
    This step cross-correlates frame-level RMS envelopes to find that delay
    and trims the matching number of samples from the front of capture.bin so
    that downstream scripts need no lag correction at all.
    """
    import numpy as np

    # ── decode audio to float32 mono via ffmpeg ───────────────────────────────
    try:
        r = subprocess.run(
            ['ffmpeg', '-v', 'quiet', '-i', audio_path,
             '-f', 'f32le', '-ac', '1', '-ar', str(aud_sr), '-'],
            capture_output=True,
        )
        audio = np.frombuffer(r.stdout, dtype=np.float32)
    except Exception as exc:
        print(f"[capture] align_lag: cannot decode audio — {exc}")
        return

    # ── load capture as float32 ───────────────────────────────────────────────
    try:
        capture = np.fromfile(capture_path, dtype=np.float32)
    except Exception as exc:
        print(f"[capture] align_lag: cannot read capture — {exc}")
        return

    if len(audio) == 0 or len(capture) == 0:
        print("[capture] align_lag: empty audio or capture — skipping")
        return

    # ── frame-level RMS for both signals ─────────────────────────────────────
    aud_frame = int(round(frame_s * aud_sr))
    cap_frame = int(round(frame_s * samp_rate))

    n_aud = len(audio)   // aud_frame
    n_cap = len(capture) // cap_frame
    n_frames = min(n_aud, n_cap)

    if n_frames < 4:
        print("[capture] align_lag: too few frames — skipping")
        return

    aud_rms = np.array([np.sqrt(np.mean(audio  [i*aud_frame:(i+1)*aud_frame]**2)) for i in range(n_frames)])
    cap_rms = np.array([np.sqrt(np.mean(capture[i*cap_frame:(i+1)*cap_frame]**2)) for i in range(n_frames)])

    # ── cross-correlate to find physical lag ─────────────────────────────────
    # np.correlate(a, c) convention: xcorr peaks at negative numpy-lag when c
    # is delayed behind a.  Physical delay (powerline lags audio) → numpy peak
    # at NEGATIVE lag.  Search [-700 ms, 0] to find powerline-delayed peaks
    # without aliasing onto the ~1.2 s repetition period of "hello×5".
    max_lag_frames = int(round(0.700 / frame_s))
    a = (aud_rms - aud_rms.mean()) / (aud_rms.std() + 1e-12)
    c = (cap_rms - cap_rms.mean()) / (cap_rms.std() + 1e-12)
    xcorr = np.correlate(a, c, mode='full') / n_frames
    lags = np.arange(-(n_frames - 1), n_frames)
    # Physical delay maps to negative numpy lag → search [-max, 0]
    valid = (-max_lag_frames <= lags) & (lags <= 0)
    numpy_lag = int(lags[valid][np.argmax(xcorr[valid])])
    # Physical lag: positive = powerline delayed behind audio
    phys_lag_frames = -numpy_lag
    lag_ms = phys_lag_frames * frame_s * 1000

    lag_path = os.path.splitext(capture_path)[0] + '.lag'
    print(f"[capture] Step 3 — lag alignment: powerline delayed {lag_ms:.0f} ms")

    if phys_lag_frames <= 0:
        # No positive physical delay — write 0 so analysis applies no shift
        try:
            open(lag_path, 'w').write("0.0\n")
        except Exception:
            pass
        print("          no physical delay detected — capture.bin unchanged, sidecar=0")
        return

    # Trim phys_lag_frames from start so cap[0] = audio[0].
    # Write 0 to sidecar AFTER trimming — capture is now aligned, analysis needs no shift.
    trim_samples = phys_lag_frames * cap_frame
    trim_bytes   = trim_samples * 4
    current_bytes = os.path.getsize(capture_path)
    if trim_bytes >= current_bytes:
        print(f"          lag trim >= file size — skipping")
        open(lag_path, 'w').write(f"{lag_ms:.1f}\n")
        return

    CHUNK = 4 * 1024 * 1024
    try:
        with open(capture_path, 'r+b') as f:
            read_pos  = trim_bytes
            write_pos = 0
            remaining = current_bytes - trim_bytes
            while remaining > 0:
                f.seek(read_pos)
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                f.seek(write_pos)
                f.write(data)
                n = len(data)
                read_pos  += n
                write_pos += n
                remaining -= n
            f.truncate(write_pos)
    except Exception as exc:
        print(f"[capture] align_lag: write error — {exc}")
        return

    final = os.path.getsize(capture_path)
    print(f"          trimmed {trim_bytes/1e6:.3f} MB ({lag_ms:.0f} ms)  →  capture.bin = {final/1e6:.3f} MB")
    # Sidecar = 0: capture is now aligned, analysis scripts need no lag shift
    try:
        open(lag_path, 'w').write("0.0\n")
    except Exception as exc:
        print(f"[capture] align_lag: cannot write lag sidecar — {exc}")


# ── flowgraph ─────────────────────────────────────────────────────────────────

class simple_rx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Powerline Leakage Monitor", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "simple_rx")
        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 200e3
        self.duration  = duration  = int(samp_rate * 20)
        self.gain      = gain      = 1.0   # normalized 0–1 (max — ADC digital gain 0–6.5 dB on this board)
        # Real RX gain range for this LFRX/Basic front-end (queried from the device).
        # The slider's dB readout uses this instead of a generic 76 dB assumption.
        self._gain_db_max  = 6.5
        # DC blocker: strip the ~0.36 ADC DC pedestal from the recorded + displayed
        # stream so the small AC ripple isn't riding on a large constant offset.
        self.dc_block      = True
        self.dc_block_taps = 1024

        ##################################################
        # Capture alignment markers
        ##################################################
        # Set by the audio thread; read by main() after qapp.exec_() returns.
        self.capture_start_bytes = None
        self.capture_end_bytes   = None

        ##################################################
        # Audio
        ##################################################
        self._audio_proc = None
        self._audio_file = find_audio_file()
        audio_label = (os.path.basename(self._audio_file)
                       if self._audio_file else "No audio file found")
        self.setWindowTitle(f"Powerline Leakage Monitor  —  {audio_label}")

        ##################################################
        # Status bar
        ##################################################
        self._status_label = Qt.QLabel(
            f"<b>Audio:</b> {audio_label} &nbsp;|&nbsp; "
            f"<b>Samp rate:</b> {int(samp_rate/1e3)} kSps &nbsp;|&nbsp; "
            f"<b>Sink:</b> capture.bin"
        )
        self._status_label.setStyleSheet(
            "QLabel { background:#1e1e2e; color:#cdd6f4; "
            "padding:4px 8px; font-family:monospace; }"
        )
        self.top_layout.insertWidget(0, self._status_label)

        ##################################################
        # Gain slider
        ##################################################
        gain_row = Qt.QWidget()
        gain_row.setStyleSheet("background:#1e1e2e;")
        gain_layout = Qt.QHBoxLayout(gain_row)
        gain_layout.setContentsMargins(8, 2, 8, 2)

        gain_lbl = Qt.QLabel("<b style='color:#cdd6f4'>RX Gain:</b>")
        gain_layout.addWidget(gain_lbl)

        self._gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self._gain_slider.setMinimum(0)
        self._gain_slider.setMaximum(100)
        self._gain_slider.setValue(int(gain * 100))
        self._gain_slider.setTickInterval(10)
        self._gain_slider.setTickPosition(Qt.QSlider.TicksBelow)
        self._gain_slider.setStyleSheet(
            "QSlider::groove:horizontal { height:6px; background:#313244; border-radius:3px; }"
            "QSlider::handle:horizontal  { width:14px; height:14px; margin:-4px 0; "
            "  background:#89b4fa; border-radius:7px; }"
            "QSlider::sub-page:horizontal { background:#89b4fa; border-radius:3px; }"
        )
        gain_layout.addWidget(self._gain_slider, stretch=1)

        self._gain_val_label = Qt.QLabel()
        self._gain_val_label.setMinimumWidth(160)
        self._gain_val_label.setStyleSheet("color:#f9e2af; font-family:monospace; font-size:11px;")
        gain_layout.addWidget(self._gain_val_label)
        self._refresh_gain_label(gain)

        self._gain_slider.valueChanged.connect(self._on_gain_slider)
        self.top_layout.insertWidget(1, gain_row)

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", "addr=192.168.10.4")),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0, 1))),
        )
        self.uhd_usrp_source_0.set_subdev_spec("A:AB", 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(0, 0)
        self.uhd_usrp_source_0.set_antenna('RXA', 0)
        self.uhd_usrp_source_0.set_bandwidth(samp_rate, 0)
        self.uhd_usrp_source_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0.set_normalized_gain(gain, 0)

        # 32768-pt Blackman FFT → ~6 Hz resolution at 200 kSps
        self.qtgui_sink_x_0 = qtgui.sink_f(
            32768,
            window.WIN_BLACKMAN,
            0,
            samp_rate,
            'Powerline Leakage — Spectrum / Waterfall / Time',
            True, True, True, True,
            None
        )
        self.qtgui_sink_x_0.set_update_time(1.0 / 10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(
            self.qtgui_sink_x_0.qwidget(), Qt.QWidget
        )
        self.qtgui_sink_x_0.enable_rf_freq(False)
        self.top_layout.addWidget(self._qtgui_sink_x_0_win)

        self.blocks_file_sink_0_0_0 = blocks.file_sink(
            gr.sizeof_float * 1, CAPTURE_PATH, False
        )
        self.blocks_file_sink_0_0_0.set_unbuffered(False)

        self.blocks_complex_to_float_0 = blocks.complex_to_float(1)

        if self.dc_block:
            # long-form ('True') DC blocker: subtracts a moving average so only
            # the DC term is removed, leaving the sub-Hz/mains content intact.
            self.blocks_dc_blocker_0 = gr_filter.dc_blocker_ff(self.dc_block_taps, True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source_0, 0),
                     (self.blocks_complex_to_float_0, 0))
        if self.dc_block:
            self.connect((self.blocks_complex_to_float_0, 0),
                         (self.blocks_dc_blocker_0, 0))
            tap = (self.blocks_dc_blocker_0, 0)
        else:
            tap = (self.blocks_complex_to_float_0, 0)
        # tap feeds both the recorder and the live scope so they match
        self.connect(tap, (self.blocks_file_sink_0_0_0, 0))
        self.connect(tap, (self.qtgui_sink_x_0, 0))

        ##################################################
        # Kick off audio after USRP settles
        ##################################################
        if self._audio_file:
            self._start_audio_thread()

    # ── audio & alignment ─────────────────────────────────────────────────────

    def _start_audio_thread(self):
        def _play():
            time.sleep(2)   # USRP settle — pre-audio samples discarded by trim

            # ── start marker: byte offset at t=0 of audio ────────────────────
            self.capture_start_bytes = _fsize(CAPTURE_PATH)
            print(f"[audio] Capture start: byte {self.capture_start_bytes} "
                  f"({self.capture_start_bytes / 4 / self.samp_rate:.3f}s into stream)")

            # Set monitor sink to 100% before playback
            subprocess.run(
                ['pactl', 'set-sink-volume', AUDIO_SINK, '100%'],
                capture_output=True,
            )
            print(f"[audio] Sink volume → 100%  ({AUDIO_SINK})")

            self._update_status("PLAYING")
            audio_env = {**os.environ, 'PULSE_SINK': AUDIO_SINK}
            self._audio_proc = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', '-volume', '100',
                 self._audio_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=audio_env,
            )
            self._audio_proc.wait()

            # ── end marker: drain pipeline then keep recording for lag tail ──
            # PIPELINE_DRAIN_S: lets GNU Radio buffers flush to the file so the
            # marker lands on the true audio boundary.
            # LAG_TAIL_S: keeps the USRP recording after audio ends so the
            # powerline's delayed response (up to ~700 ms) is fully captured.
            # align_lag() (Step 3) will trim the matching lead from the front.
            time.sleep(PIPELINE_DRAIN_S + LAG_TAIL_S)
            self.capture_end_bytes = _fsize(CAPTURE_PATH)

            captured_s = ((self.capture_end_bytes - self.capture_start_bytes)
                          / 4 / self.samp_rate)
            print(f"[audio] Playback done — ~{captured_s:.3f}s captured "
                  f"(bytes {self.capture_start_bytes}–{self.capture_end_bytes})")
            self._update_status("DONE")

            # Exit the Qt event loop; main() will stop the flowgraph and trim.
            Qt.QApplication.quit()

        threading.Thread(target=_play, daemon=True).start()

    def _update_status(self, state):
        audio_label = os.path.basename(self._audio_file) if self._audio_file else "—"
        color = {"PLAYING": "#a6e3a1", "DONE": "#f38ba8"}.get(state, "#cdd6f4")
        self._status_label.setText(
            f"<b>Audio:</b> <span style='color:{color}'>{state}</span> "
            f"{audio_label} &nbsp;|&nbsp; "
            f"<b>Samp rate:</b> {int(self.samp_rate/1e3)} kSps &nbsp;|&nbsp; "
            f"<b>Sink:</b> capture.bin"
        )

    # ── Qt lifecycle ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # Stop audio if window is closed manually before audio finishes
        if self._audio_proc and self._audio_proc.poll() is None:
            self._audio_proc.terminate()
            try:
                self._audio_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._audio_proc.kill()
        self.settings = Qt.QSettings("GNU Radio", "simple_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        # NOTE: do NOT call stop()/wait() or _trim_capture() here.
        # main() does that after qapp.exec_() returns, which fires whether the
        # app exits via QApplication.quit() or via the user closing the window.
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_duration(int(self.samp_rate * 20))
        self.qtgui_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_bandwidth(self.samp_rate, 0)

    def get_duration(self):
        return self.duration

    def set_duration(self, duration):
        self.duration = duration

    # ── gain control ──────────────────────────────────────────────────────────

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = float(gain)
        self.uhd_usrp_source_0.set_normalized_gain(self.gain, 0)
        self._refresh_gain_label(self.gain)

    def _refresh_gain_label(self, norm_gain):
        """Update the gain value label next to the slider."""
        # dB from the real device gain range (LFRX/Basic: 0–6.5 dB, digital ADC gain)
        db_approx = norm_gain * self._gain_db_max
        self._gain_val_label.setText(
            f"{norm_gain:.2f}  (~{db_approx:.0f} dB)  "
            f"{'▮' * int(norm_gain * 20)}{'▯' * (20 - int(norm_gain * 20))}"
        )

    def _on_gain_slider(self, value):
        """Slider moved — update USRP gain in real time."""
        self.set_gain(value / 100.0)


# ── entry point ───────────────────────────────────────────────────────────────

def main(top_block_cls=simple_rx, options=None):
    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()   # blocks until QApplication.quit() or window close

    # ── post-run: stop flowgraph, flush file sink, trim capture ──────────────
    # This runs whether the app exited via audio-auto-quit, window close, or
    # Ctrl-C (SIGINT → sig_handler → QApplication.quit()).
    tb.stop()
    tb.wait()   # blocks until file_sink has flushed all pending writes

    trim_capture(
        CAPTURE_PATH,
        tb.capture_start_bytes,
        tb.capture_end_bytes,
        tb.samp_rate,
        audio_duration_s=get_audio_duration(tb._audio_file),
    )
    align_lag(CAPTURE_PATH, tb._audio_file, tb.samp_rate)

if __name__ == '__main__':
    main()
