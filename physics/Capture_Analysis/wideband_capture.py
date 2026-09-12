#!/usr/bin/env python3
"""
wideband_capture.py — High-rate USRP capture to hunt for a Class-D / SMPS
                      switching carrier on the soundbar's AC-cord current.

The existing simple_rx.grc samples at 200 kSps → only 0-100 kHz is visible, and
we confirmed all energy there is the <1 kHz PSU/mains envelope (no carrier).
A Class-D amplifier's PWM switching frequency is typically 250 kHz – 1 MHz and
carries the audio as wideband AM sidebands — but it lives ABOVE your current
100 kHz Nyquist.  This script samples fast enough (default 4 MSps → 2 MHz of
bandwidth) to expose it, writing the real ADC stream to a .bin file exactly like
simple_rx (subdev A:AB, antenna RXA, real part to disk).

Run this WHILE playing audio on the soundbar (pass --audio to auto-play a file
after the front-end settles), then analyse with wideband_survey.py.

Hardware notes
--------------
  * Same USRP/daughterboard as simple_rx (addr=192.168.10.4, subdev A:AB, RXA).
  * On a real-sampling LF/Basic board, port 0 (real / ADC-A) IS your probe
    signal — we keep it and ignore the unconnected Q, matching your setup.
  * File size = samp_rate × 4 bytes/s.  4 MSps × 15 s ≈ 240 MB.  Keep it short.
  * The current probe / clamp must have bandwidth up to the carrier; if it rolls
    off below ~300 kHz you may see nothing even if a carrier exists — that is
    itself a useful (probe-limited) result.

Usage
-----
    python3 wideband_capture.py --samp-rate 4e6 --duration 15 --gain 0.8 \\
                                --audio hello5.wav --out wideband.bin
    python3 wideband_capture.py --samp-rate 8e6 --duration 10   # widest scan
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'wideband.bin')


def play_audio_later(audio_path, delay_s):
    """Start playback after the USRP settles, on a background thread."""
    player = None
    for cand in ('ffplay', 'aplay', 'paplay'):
        if shutil.which(cand):
            player = cand
            break
    if player is None:
        print('[audio] no ffplay/aplay/paplay found — play audio manually NOW')
        return

    def _run():
        time.sleep(delay_s)
        if player == 'ffplay':
            cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', audio_path]
        else:
            cmd = [player, audio_path]
        print(f'[audio] playing {os.path.basename(audio_path)} via {player}')
        subprocess.run(cmd)

    threading.Thread(target=_run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samp-rate', type=float, default=4e6,
                    help='Sample rate in Sps (default 4e6 → 2 MHz bandwidth)')
    ap.add_argument('--duration', type=float, default=15.0,
                    help='Capture duration in seconds (default 15)')
    ap.add_argument('--gain', type=float, default=0.8,
                    help='Normalized RX gain 0-1 (default 0.8)')
    ap.add_argument('--freq', type=float, default=0.0,
                    help='Center frequency Hz (default 0 = baseband real sampling)')
    ap.add_argument('--addr', default='192.168.10.4', help='USRP IP address')
    ap.add_argument('--subdev', default='A:AB', help='Subdev spec (default A:AB)')
    ap.add_argument('--antenna', default='RXA', help='Antenna port (default RXA)')
    ap.add_argument('--out', default=DEFAULT_OUT, help='Output .bin (real float32)')
    ap.add_argument('--audio', default=None, help='Audio file to auto-play during capture')
    ap.add_argument('--settle', type=float, default=2.0,
                    help='Front-end settle delay before audio/capture (s, default 2)')
    ap.add_argument('--dc-block', action='store_true',
                    help='Insert a DC blocker (removes the ~0.2 DC pedestal so gain '
                         'can use the ADC range for the small AC ripple)')
    args = ap.parse_args()

    try:
        from gnuradio import gr, uhd, blocks
    except ImportError as e:
        sys.exit(f'[error] GNU Radio not importable here: {e}\n'
                 f'        Run this on the capture machine with gr-uhd installed.')

    samp_rate = args.samp_rate
    n_samps = int(round(args.duration * samp_rate))

    print(f'[cap]   requesting {samp_rate/1e6:.3f} MSps  → {samp_rate/2e6:.3f} MHz band')
    print(f'[cap]   {args.duration:.1f}s  = {n_samps:,} samples  '
          f'≈ {n_samps*4/1e6:.0f} MB')

    tb = gr.top_block('wideband_capture')

    src = uhd.usrp_source(
        ','.join(('', f'addr={args.addr}')),
        uhd.stream_args(cpu_format='fc32', args='', channels=[0]),
    )
    src.set_subdev_spec(args.subdev, 0)
    src.set_samp_rate(samp_rate)
    src.set_center_freq(args.freq, 0)
    src.set_antenna(args.antenna, 0)
    src.set_bandwidth(samp_rate, 0)
    src.set_rx_agc(False, 0)
    src.set_normalized_gain(args.gain, 0)

    actual = src.get_samp_rate()
    print(f'[cap]   actual samp_rate = {actual/1e6:.4f} MSps  '
          f'(Nyquist {actual/2e6:.3f} MHz)')
    n_samps = int(round(args.duration * actual))

    c2f = blocks.complex_to_float(1)                 # port 0 = real (ADC-A)
    head = blocks.head(gr.sizeof_float, n_samps)
    sink = blocks.file_sink(gr.sizeof_float, args.out, False)
    sink.set_unbuffered(False)

    tb.connect((src, 0), (c2f, 0))
    if args.dc_block:
        dcb = blocks.dc_blocker_ff(1024, True)
        tb.connect((c2f, 0), (dcb, 0))
        tb.connect((dcb, 0), (head, 0))
        print('[cap]   DC blocker enabled (1024-tap)')
    else:
        tb.connect((c2f, 0), (head, 0))
    tb.connect((head, 0), (sink, 0))

    if args.audio and os.path.exists(args.audio):
        play_audio_later(args.audio, args.settle)
    elif args.audio:
        print(f'[audio] not found: {args.audio}')
    else:
        print(f'[audio] no --audio: start playback on the soundbar NOW '
              f'(capturing for {args.duration:.0f}s after {args.settle:.0f}s settle)')

    time.sleep(args.settle)
    print('[cap]   recording …')
    t0 = time.time()
    tb.run()                       # blocks until head passes n_samps
    tb.stop(); tb.wait()
    dt = time.time() - t0
    sz = os.path.getsize(args.out)
    print(f'[cap]   done in {dt:.1f}s  →  {args.out}  ({sz/1e6:.0f} MB)')
    print()
    print(f'  Next: python3 wideband_survey.py --capture {os.path.basename(args.out)} '
          f'--samp-rate {actual:.0f} --audio {os.path.basename(args.audio) if args.audio else "<played.wav>"}')


if __name__ == '__main__':
    main()
