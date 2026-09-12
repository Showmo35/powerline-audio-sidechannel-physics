"""
Generate a 3-minute MP3 containing 60 three-second tones sweeping from 20 Hz to 20 kHz.
Requires: numpy, pydub, and an available ffmpeg encoder for MP3 export.
"""

import argparse
from pathlib import Path
import shutil
import numpy as np
from pydub import AudioSegment


SAMPLE_RATE = 22050
DURATION_PER_TONE = 3.0  # seconds
TOTAL_DURATION = 180.0  # seconds
START_FREQ = 20.0  # Hz
END_FREQ = 20_000.0  # Hz
BITRATE = "192k"
CHIRP_DURATION = 20.0  # seconds per chirp
CHIRP_GAP = 3.0  # seconds of silence between chirps


def generate_tone(frequency: float, duration: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.5) -> AudioSegment:
    """Create a mono sine wave tone segment at the given frequency, duration, and amplitude."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t)  # adjustable amplitude
    samples = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    return AudioSegment(data=samples.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)


def build_sweep(start_freq: float, end_freq: float, total_duration: float, duration_per_tone: float) -> AudioSegment:
    """Construct the full sweep as back-to-back tone segments."""
    num_tones = int(total_duration // duration_per_tone)
    frequencies = np.linspace(start_freq, end_freq, num=num_tones)
    segments = [generate_tone(freq, duration_per_tone) for freq in frequencies]
    return sum(segments)


def build_stepped_tones_with_gaps(start_freq: float, end_freq: float, step_freq: float, tone_duration: float, gap_duration: float) -> AudioSegment:
    """Generate tones at specific frequency intervals with gaps between them."""
    # Generate frequency list from start to end with step interval
    frequencies = np.arange(start_freq, end_freq + step_freq, step_freq)
    
    segments = []
    gap = AudioSegment.silent(duration=int(gap_duration * 1000), frame_rate=SAMPLE_RATE)
    
    for freq in frequencies:
        tone = generate_tone(freq, tone_duration)
        segments.append(tone)
        segments.append(gap)
    
    # Remove the last gap
    if segments:
        segments = segments[:-1]
    
    return sum(segments) if segments else AudioSegment.empty()


def build_amplitude_sweep(frequency: float, total_duration: float, tone_duration: float, gap_duration: float, min_amplitude: float = 0.1, max_amplitude: float = 0.9) -> AudioSegment:
    """Generate tones at the same frequency with increasing amplitude over time."""
    cycle_duration = tone_duration + gap_duration
    num_cycles = int(total_duration // cycle_duration)
    
    # Generate amplitude values that increase linearly
    amplitudes = np.linspace(min_amplitude, max_amplitude, num_cycles)
    
    segments = []
    gap = AudioSegment.silent(duration=int(gap_duration * 1000), frame_rate=SAMPLE_RATE)
    
    for i, amplitude in enumerate(amplitudes):
        tone = generate_tone(frequency, tone_duration, amplitude=amplitude)
        segments.append(tone)
        if i < len(amplitudes) - 1:  # Don't add gap after the last tone
            segments.append(gap)
    
    return sum(segments) if segments else AudioSegment.empty()


def build_continuous_amplitude_sweep(frequency: float, total_duration: float, min_amplitude: float = 0.1, max_amplitude: float = 0.9, sample_rate: int = SAMPLE_RATE) -> AudioSegment:
    """Generate a continuous tone with smoothly varying amplitude over time."""
    t = np.linspace(0, total_duration, int(total_duration * sample_rate), endpoint=False)
    
    # Create smooth amplitude envelope that increases linearly over time
    amplitude_envelope = np.linspace(min_amplitude, max_amplitude, len(t))
    
    # Generate the sine wave with varying amplitude
    waveform = amplitude_envelope * np.sin(2 * np.pi * frequency * t)
    
    # Convert to audio samples
    samples = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    
    return AudioSegment(data=samples.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)


def generate_chirp(start_freq: float, end_freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> AudioSegment:
    """Create a mono linear chirp from start_freq to end_freq over the given duration."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    k = (end_freq - start_freq) / duration  # rate of frequency change (Hz per second)
    phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
    waveform = 0.5 * np.sin(phase)
    samples = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    return AudioSegment(data=samples.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)


def build_repeating_chirp(
    total_duration: float,
    chirp_duration: float,
    gap_duration: float,
    start_freq: float,
    end_freq: float,
) -> AudioSegment:
    """Repeat chirp + gap until reaching total_duration, then trim to exact length."""
    chirp = generate_chirp(start_freq, end_freq, chirp_duration)
    gap = AudioSegment.silent(duration=int(gap_duration * 1000), frame_rate=SAMPLE_RATE)
    pieces = []
    current_ms = 0
    total_ms = int(total_duration * 1000)
    per_cycle_ms = len(chirp) + len(gap)

    while current_ms + per_cycle_ms <= total_ms:
        pieces.append(chirp)
        pieces.append(gap)
        current_ms += per_cycle_ms

    if current_ms < total_ms:
        pieces.append(chirp)

    combined = sum(pieces)
    return combined[:total_ms]


def build_varying_duration_chirps(
    start_freq: float,
    end_freq: float,
    num_chirps: int = 10,
    max_duration: float = 8.0,
    gap_duration: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
) -> AudioSegment:
    """
    Generate chirps with varying durations from max_duration down to max_duration/num_chirps.
    Each chirp sweeps from start_freq to end_freq, with gaps between them.
    
    Args:
        start_freq: Starting frequency (Hz)
        end_freq: Ending frequency (Hz)
        num_chirps: Number of chirps to generate (default: 10)
        max_duration: Duration of the first chirp in seconds (default: 8.0)
        gap_duration: Gap duration between chirps in seconds (default: 3.0)
        sample_rate: Sample rate for audio generation
    
    Returns:
        AudioSegment containing all chirps with gaps
    """
    pieces = []
    gap = AudioSegment.silent(duration=int(gap_duration * 1000), frame_rate=sample_rate)
    
    for i in range(num_chirps):
        # Calculate duration for this chirp: from max_duration down to max_duration/num_chirps
        duration = max_duration - (i * max_duration / num_chirps)
        
        # Generate chirp at current sample rate
        chirp = generate_chirp(start_freq, end_freq, duration, sample_rate)
        pieces.append(chirp)
        
        # Add gap after each chirp except the last one
        if i < num_chirps - 1:
            pieces.append(gap)
    
    return sum(pieces) if pieces else AudioSegment.empty()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create audio sweeps: tones stepping 20 Hz-20 kHz, stepped tones 1-20 kHz, or repeating chirps 0-20 kHz."
    )
    parser.add_argument(
        "--pattern",
        choices=["tones", "stepped", "chirp", "amplitude", "continuous", "varying-chirp"],
        default="tones",
        help="tones: 3-second steps across 20Hz-20kHz; stepped: 1-20kHz with 2kHz steps, 3s tone + 3s gap; chirp: 20s chirp 0-20 kHz with 3s gaps; amplitude: same frequency with increasing amplitude for 90s; continuous: smooth continuous amplitude sweep; varying-chirp: 10 chirps 0-5kHz with varying durations (8s to 0.8s) and 3s gaps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tone_sweep.mp3"),
        help="Output MP3 file path (default: tone_sweep.mp3)",
    )
    parser.add_argument("--chirp-duration", type=float, default=CHIRP_DURATION, help="Seconds per chirp segment")
    parser.add_argument("--chirp-gap", type=float, default=CHIRP_GAP, help="Seconds of silence between chirps")
    parser.add_argument("--tone-duration", type=float, default=3.0, help="Duration of each tone in stepped pattern (seconds)")
    parser.add_argument("--gap-duration", type=float, default=3.0, help="Duration of gap between tones in stepped pattern (seconds)")
    parser.add_argument("--amplitude-freq", type=float, default=1000.0, help="Frequency for amplitude sweep (Hz)")
    parser.add_argument("--amplitude-duration", type=float, default=90.0, help="Total duration for amplitude sweep (seconds)")
    parser.add_argument("--min-amplitude", type=float, default=0.1, help="Minimum amplitude (0.0-1.0)")
    parser.add_argument("--max-amplitude", type=float, default=0.9, help="Maximum amplitude (0.0-1.0)")
    parser.add_argument("--varying-chirp-freq", type=float, default=5000.0, help="End frequency for varying-duration chirps (Hz)")
    parser.add_argument("--num-chirps", type=int, default=10, help="Number of chirps in varying-chirp pattern")
    parser.add_argument("--max-chirp-duration", type=float, default=8.0, help="Maximum chirp duration in varying-chirp pattern (seconds)")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg is required to export MP3. Install it (e.g., `conda install -c conda-forge ffmpeg`) "
            "and rerun the script."
        )

    if args.pattern == "tones":
        sweep = build_sweep(START_FREQ, END_FREQ, TOTAL_DURATION, DURATION_PER_TONE)
    elif args.pattern == "stepped":
        sweep = build_stepped_tones_with_gaps(
            start_freq=1000.0,  # 1 kHz
            end_freq=20000.0,   # 20 kHz
            step_freq=2000.0,   # 2 kHz intervals
            tone_duration=args.tone_duration,
            gap_duration=args.gap_duration
        )
    elif args.pattern == "amplitude":
        sweep = build_amplitude_sweep(
            frequency=args.amplitude_freq,
            total_duration=args.amplitude_duration,
            tone_duration=args.tone_duration,
            gap_duration=args.gap_duration,
            min_amplitude=args.min_amplitude,
            max_amplitude=args.max_amplitude
        )
    elif args.pattern == "continuous":
        sweep = build_continuous_amplitude_sweep(
            frequency=args.amplitude_freq,
            total_duration=args.amplitude_duration,
            min_amplitude=args.min_amplitude,
            max_amplitude=args.max_amplitude
        )
    elif args.pattern == "varying-chirp":
        # Generate for both 22.05 kHz and 44.1 kHz sample rates
        # First, generate the 22.05 kHz version
        sweep_22050 = build_varying_duration_chirps(
            start_freq=0.0,
            end_freq=args.varying_chirp_freq,
            num_chirps=args.num_chirps,
            max_duration=args.max_chirp_duration,
            gap_duration=CHIRP_GAP,
            sample_rate=22050
        )
        
        # Generate the 44.1 kHz version
        sweep_44100 = build_varying_duration_chirps(
            start_freq=0.0,
            end_freq=args.varying_chirp_freq,
            num_chirps=args.num_chirps,
            max_duration=args.max_chirp_duration,
            gap_duration=CHIRP_GAP,
            sample_rate=44100
        )
        
        # Export both versions
        output_base = args.output.stem
        output_dir = args.output.parent
        
        output_22050 = output_dir / f"{output_base}_22050Hz.mp3"
        output_44100 = output_dir / f"{output_base}_44100Hz.mp3"
        
        sweep_22050.export(output_22050, format="mp3", bitrate=BITRATE)
        sweep_44100.export(output_44100, format="mp3", bitrate=BITRATE)
        
        print(f"Saved 22.05 kHz sweep to {output_22050.resolve()}")
        print(f"Saved 44.1 kHz sweep to {output_44100.resolve()}")
        
        # Calculate durations for each chirp
        chirp_durations = [args.max_chirp_duration - (i * args.max_chirp_duration / args.num_chirps) 
                          for i in range(args.num_chirps)]
        total_time = sum(chirp_durations) + (args.num_chirps - 1) * CHIRP_GAP
        
        print(f"Generated {args.num_chirps} chirps from 0 Hz to {args.varying_chirp_freq} Hz")
        print(f"Chirp durations: {args.max_chirp_duration:.1f}s down to {args.max_chirp_duration/args.num_chirps:.2f}s")
        print(f"Gap between chirps: {CHIRP_GAP}s")
        print(f"Total duration: {total_time:.1f}s")
        print(f"Sample rates: 22050 Hz and 44100 Hz")
        return  # Exit early since we've already saved both files
    else:  # chirp
        sweep = build_repeating_chirp(
            total_duration=TOTAL_DURATION,
            chirp_duration=args.chirp_duration,
            gap_duration=args.chirp_gap,
            start_freq=0.0,
            end_freq=END_FREQ,
        )

    sweep.export(args.output, format="mp3", bitrate=BITRATE)
    print(f"Saved sweep to {args.output.resolve()}")
    
    if args.pattern == "stepped":
        num_tones = int((20000 - 1000) / 2000) + 1
        total_time = num_tones * (args.tone_duration + args.gap_duration) - args.gap_duration
        print(f"Generated {num_tones} tones from 1-20 kHz (2 kHz steps)")
        print(f"Each tone: {args.tone_duration}s, gap: {args.gap_duration}s")
        print(f"Total duration: {total_time:.1f}s")
    elif args.pattern == "amplitude":
        cycle_duration = args.tone_duration + args.gap_duration
        num_cycles = int(args.amplitude_duration // cycle_duration)
        actual_duration = num_cycles * cycle_duration - args.gap_duration
        print(f"Generated {num_cycles} amplitude steps at {args.amplitude_freq} Hz")
        print(f"Amplitude range: {args.min_amplitude:.2f} - {args.max_amplitude:.2f}")
        print(f"Each tone: {args.tone_duration}s, gap: {args.gap_duration}s")
        print(f"Total duration: {actual_duration:.1f}s")
    elif args.pattern == "continuous":
        print(f"Generated continuous amplitude sweep at {args.amplitude_freq} Hz")
        print(f"Amplitude range: {args.min_amplitude:.2f} - {args.max_amplitude:.2f}")
        print(f"Smooth continuous sweep over {args.amplitude_duration:.1f}s")


if __name__ == "__main__":
    main()
