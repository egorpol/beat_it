import librosa
import numpy as np
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import librosa.display
from bokeh.plotting import figure
from bokeh.models import Span, LinearColorMapper, ColorBar, Range1d

# Human-readable labels for feature plots
FEATURE_LABELS = {
    'waveform': 'Waveform',
    'stft_db': 'STFT (dB)',
    'melspectrogram': 'Mel Spectrogram (dB)',
    'amplitude_envelope': 'Amplitude Envelope (RMS)',
    'zero_crossing_rate': 'Zero-Crossing Rate',
    'spectral_centroid': 'Spectral Centroid',
    'spectral_flux': 'Spectral Flux',
    'high_frequency_content': 'High-Frequency Content',
    'spectral_contrast': 'Spectral Contrast',
    'cqt_flux': 'CQT Onset Novelty',
    'rms_low': 'Low-Band RMS',
    'rms_mid': 'Mid-Band RMS',
    'rms_high': 'High-Band RMS',
    'phase_flux': 'Phase Flux'
}

# --- add imports near the top ---
import re
from typing import Tuple

# --- time parsing helpers ---
_FALLBACK_FPS = 30.0
_FPS_ALIAS = {
    '23.976': 24000/1001,
    '29.97':  30000/1001,
    '59.94':  60000/1001,
    '75':     75.0,     # CD-DA
}
_RANGE_SPLIT = re.compile(r'\s*[-–—]\s*')  # hyphen / en dash / em dash

# SMPTE HH:MM:SS:FF or HH:MM:SS;FF with optional @fps / fps suffix
_TC_RE = re.compile(
    r'^\s*(?P<h>\d{1,2})[:;](?P<m>\d{2})[:;](?P<s>\d{2})[:;](?P<f>\d{2})'
    r'(?:\s*@\s*(?P<fps>[\d.]+)\s*(?:fps)?)?\s*$',
    re.IGNORECASE
)
# Clock HH:MM:SS(.mmm) or MM:SS(.mmm)
_CLOCK_RE = re.compile(
    r'^\s*(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:[.,]\d+)?)\s*$'
)
# Plain seconds (optionally with unit)
_SEC_RE = re.compile(r'^\s*(?P<s>\d+(?:[.,]\d+)?)\s*(?:s|sec|seconds?)?\s*$', re.IGNORECASE)

def _parse_fps(txt: str | None) -> float:
    if not txt:
        return _FALLBACK_FPS
    s = txt.lower().replace('fps', '').replace('df', '').strip()
    if s in _FPS_ALIAS:
        return float(_FPS_ALIAS[s])
    try:
        return float(s)
    except Exception:
        return _FALLBACK_FPS

def parse_timestamp(ts: str, default_fps: float = _FALLBACK_FPS) -> float:
    """Return seconds from a time string."""
    t = ts.strip()
    m = _TC_RE.match(t)
    if m:
        fps = _parse_fps(m.group('fps')) or default_fps
        h = int(m['h']); mi = int(m['m']); s = int(m['s']); f = int(m['f'])
        return h*3600 + mi*60 + s + f / fps

    m = _CLOCK_RE.match(t.replace(';', ':'))
    if m:
        h = int(m.group('h') or 0)
        mi = int(m.group('m'))
        s = float(m.group('s').replace(',', '.'))
        return h*3600 + mi*60 + s

    m = _SEC_RE.match(t)
    if m:
        return float(m['s'].replace(',', '.'))

    raise ValueError(f"Unrecognized time format: {ts!r}")

def parse_time_range(spec: str, default_fps: float = _FALLBACK_FPS) -> Tuple[float, float | None]:
    """
    Parse 'start-end' into (offset, duration) in seconds.
    If only 'start' is given, duration is None (to EOF).
    """
    parts = _RANGE_SPLIT.split(spec.strip(), maxsplit=1)
    start = parse_timestamp(parts[0], default_fps=default_fps)
    if len(parts) == 1 or not parts[1]:
        return start, None
    end = parse_timestamp(parts[1], default_fps=default_fps)
    if end < start:
        raise ValueError(f"End before start in range {spec!r}")
    return start, end - start


def load_audio(path: str,
               sr: Optional[int] = None,
               offset: float = 0.0,
               duration: Optional[float] = None,
               time_range: Optional[str] = None) -> tuple[np.ndarray, int]:
    """
    Load audio (optionally a slice).
    - time_range: 'start-end' using HH:MM:SS.mmm, SMPTE HH:MM:SS:FF[@fps], or CD-DA MM:SS:FF@75.
      Examples: '00:10-00:20', '01:02:03.250-01:02:13.250', '00:00:04:12@25-00:00:14:00@25'
    - If provided, time_range overrides offset/duration.
    """
    if time_range:
        offset, duration = parse_time_range(time_range)

    # Try to get total duration without decoding
    try:
        total = librosa.get_duration(path=path)
    except Exception:
        total = None

    y, sr = librosa.load(path, sr=sr, offset=offset, duration=duration)
    seg_dur = librosa.get_duration(y=y, sr=sr)

    if total is not None and (offset > 0 or duration is not None):
        print(f"Loaded '{path}' -> segment {offset:.3f}–{offset+seg_dur:.3f}s of {total:.3f}s @ {sr}Hz")
    else:
        print(f"Loaded '{path}' -> {seg_dur:.3f}s @ {sr}Hz")
    return y, sr


def extract_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 128,
    compute_phase_flux: bool = True,
) -> Dict[str, Any]:
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    stft_db = librosa.amplitude_to_db(mag, ref=np.max)

    rms = librosa.feature.rms(S=mag, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr, hop_length=hop_length)[0]
    spec_flux = librosa.onset.onset_strength(S=mag, sr=sr, hop_length=hop_length)
    hfc = librosa.onset.onset_strength(
        S=mag, sr=sr, hop_length=hop_length,
        aggregate=np.mean, fmin=sr/2, n_mels=n_mels
    )
    mel = librosa.feature.melspectrogram(
        S=mag**2, sr=sr,
        n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    contrast = librosa.feature.spectral_contrast(S=mag, sr=sr, hop_length=hop_length)
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        n_bins=84, bins_per_octave=12
    ))
    cqt_flux = librosa.onset.onset_strength(S=cqt, sr=sr, hop_length=hop_length)

    n_bins = mag.shape[0]
    thirds = n_bins // 3
    rms_low = np.sqrt(np.mean(mag[:thirds, :]**2, axis=0))
    rms_mid = np.sqrt(np.mean(mag[thirds:2*thirds, :]**2, axis=0))
    rms_high = np.sqrt(np.mean(mag[2*thirds:, :]**2, axis=0))

    features: Dict[str, Any] = {
        'waveform': y,
        'stft_db': stft_db,
        'melspectrogram': mel_db,
        'amplitude_envelope': rms,
        'zero_crossing_rate': zcr,
        'spectral_centroid': centroid,
        'spectral_flux': spec_flux,
        'high_frequency_content': hfc,
        'spectral_contrast': contrast,
        'cqt_flux': cqt_flux,
        'rms_low': rms_low,
        'rms_mid': rms_mid,
        'rms_high': rms_high
    }

    if compute_phase_flux:
        phase = np.unwrap(np.angle(stft), axis=0)
        phase_diff = np.diff(phase, axis=1)
        phase_flux = np.pad(np.sum(np.clip(phase_diff, 0, None), axis=0), (1, 0), 'constant')
        features['phase_flux'] = phase_flux

    return features


def detect_onsets(
    audio_or_feats: Union[np.ndarray, Dict[str, Any]],
    sr: int,
    hop_length: int = 512,
    delta: float = 0.1,
    backtrack: bool = True,
) -> np.ndarray:
    feats = audio_or_feats if isinstance(audio_or_feats, dict) else \
            extract_features(audio_or_feats, sr, hop_length, compute_phase_flux=False)
    envelope = feats['spectral_flux']
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=sr, hop_length=hop_length,
        delta=delta, backtrack=backtrack
    )
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

def visualize_features(
    features: Dict[str, Any],
    sr: int,
    hop_length: int,
    onset_times: Optional[Union[np.ndarray, list, dict]] = None,
    features_to_plot: List[str] = None,
    backend: str = 'matplotlib',
    *,
    time_range: Optional[str] = None,
    t_start: Optional[float] = None,
    onsets_are_absolute: bool = False,
    grayscale: bool = False,
    grayscale_cmap: str = 'gray',
    font_family: Optional[str] = None,
    font_scale: float = 1.0,
    dpi: Optional[int] = None,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None
) -> None:
    # Derive t_start automatically if not provided
    if t_start is None:
        t_start = parse_time_range(time_range)[0] if time_range else 0.0

    # Shift onsets if they are relative to the slice
    def _shift_onsets(ot):
        if ot is None or onsets_are_absolute or not t_start:
            return ot
        if isinstance(ot, (list, np.ndarray)):
            return np.asarray(ot, dtype=float) + t_start
        shifted = {}
        for k, v in ot.items():
            vv = dict(v)
            vv['times'] = np.asarray(v['times'], dtype=float) + t_start
            shifted[k] = vv
        return shifted

    onset_times = _shift_onsets(onset_times)

    if backend.lower() == 'matplotlib':
        _plot_matplotlib(
            features,
            sr,
            hop_length,
            onset_times,
            features_to_plot,
            t_start,
            grayscale=grayscale,
            grayscale_cmap=grayscale_cmap,
            font_family=font_family,
            font_scale=font_scale,
            dpi=dpi,
            figsize=figsize,
            save_path=save_path,
        )
    elif backend.lower() == 'bokeh':
        from bokeh.io import show
        from bokeh.layouts import column
        plots = _plot_bokeh(
            features,
            sr,
            hop_length,
            onset_times,
            features_to_plot,
            t_start,
            grayscale=grayscale,
            font_scale=font_scale,
        )
        show(column(*plots))
    else:
        raise ValueError(f"Unknown backend: {backend}")



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import librosa
import librosa.display
from typing import Dict, Any, Optional, Union, List

def _plot_matplotlib(
    features: Dict[str, Any],
    sr: int,
    hop_length: int,
    onset_times: Optional[Union[np.ndarray, list, dict]],
    features_to_plot: List[str],
    t_start: float
) -> None:
    n = len(features_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    seg_duration = librosa.get_duration(y=features['waveform'], sr=sr)

    global_legend_handles = []
    if isinstance(onset_times, dict):
        for label, data_dict in onset_times.items():
            color = data_dict.get('color', 'r')
            proxy = Line2D([0], [0], linestyle='--', color=color, label=label)
            global_legend_handles.append(proxy)

    for ax, name in zip(axes, features_to_plot):
        data = features[name]
        if data.ndim == 1:
            if name == 'waveform':
                arr = data[::max(1, len(data)//10000)]
                times = t_start + np.linspace(0, seg_duration, len(arr))
            else:
                times = t_start + librosa.times_like(data, sr=sr, hop_length=hop_length)
                arr = data
            ax.plot(times, arr)
            ax.set_ylabel(FEATURE_LABELS.get(name, name))
        else:
            # Time-aligned spectrogram using explicit x_coords
            frame_times = t_start + librosa.times_like(data, sr=sr, hop_length=hop_length)
            y_axis = 'mel' if 'melspectrogram' in name else 'linear'
            librosa.display.specshow(
                data, sr=sr, hop_length=hop_length,
                x_axis='time', y_axis=y_axis, ax=ax, x_coords=frame_times
            )
            ax.set_ylabel('Freq' + (' (Hz)' if y_axis == 'linear' else ' (Mel)'))

        if onset_times is not None:
            ymin, ymax = ax.get_ylim()
            if isinstance(onset_times, (list, np.ndarray)):
                ax.vlines(onset_times, ymin, ymax, color='r', linestyle='--')
            else:
                for onset_data in onset_times.values():
                    ax.vlines(onset_data['times'], ymin, ymax,
                              color=onset_data.get('color', 'r'), linestyle='--')

    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Time (s)')

    if global_legend_handles:
        fig.legend(handles=global_legend_handles, loc='upper right', borderaxespad=1.0)

    plt.show()



from typing import Dict, Any, List, Optional, Union
import numpy as np
import librosa
from bokeh.plotting import figure
from bokeh.models import Span
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.models import Legend, LegendItem, Range1d

def _plot_bokeh(
    features: Dict[str, Any],
    sr: int,
    hop_length: int,
    onset_times: Optional[Union[np.ndarray, list, dict]],
    features_to_plot: List[str],
    t_start: float
) -> List[Any]:
    plots: List[Any] = []
    seg_duration = librosa.get_duration(y=features['waveform'], sr=sr)
    tools = 'pan,wheel_zoom,box_zoom,reset,save'

    for i, name in enumerate(features_to_plot):
        if i == 0:
            p = figure(width=1400, height=400, x_axis_label='Time (s)', tools=tools)
        else:
            p = figure(width=1400, height=400, x_axis_label='Time (s)', tools=tools,
                       x_range=plots[0].x_range)
        label = FEATURE_LABELS.get(name, name.replace('_', ' ').title())
        p.title.text = label

        data = features[name]
        if data.ndim == 1:
            arr = data[::max(1, len(data)//5000)] if name == 'waveform' else data
            times = (t_start + np.linspace(0, seg_duration, len(arr))
                     if name == 'waveform'
                     else t_start + librosa.times_like(data, sr=sr, hop_length=hop_length))
            p.line(times, arr, line_width=1)
            p.yaxis.axis_label = label
        else:
            mapper = LinearColorMapper(palette='Viridis256', low=np.min(data), high=np.max(data))
            if 'melspectrogram' in name:
                y0, dh = 0, data.shape[0]
                p.y_range = Range1d(y0, dh)
                p.yaxis.axis_label = 'Mel bins'
            else:
                y0, dh = 0, sr/2
                p.y_range = Range1d(y0, dh)
                p.yaxis.axis_label = 'Frequency (Hz)'
            # place image starting at t_start, width=seg_duration
            p.image(image=[data], x=t_start, y=y0, dw=seg_duration, dh=dh, color_mapper=mapper)
            p.add_layout(ColorBar(color_mapper=mapper, title='dB'), 'right')

        if onset_times is not None:
            if isinstance(onset_times, (list, np.ndarray)):
                for t in onset_times:
                    p.add_layout(Span(location=t, dimension='height', line_color='red', line_dash='dashed'))
                if i == 0:
                    p.line([t_start + seg_duration + 1], [0], line_color='red', line_dash='dashed',
                           visible=False, legend_label='Onsets')
            elif isinstance(onset_times, dict):
                for onset_label, onset_data in onset_times.items():
                    color = onset_data.get('color', 'red')
                    for t in onset_data['times']:
                        p.add_layout(Span(location=t, dimension='height', line_color=color, line_dash='dashed'))
                    if i == 0:
                        p.line([t_start + seg_duration + 1], [0], line_color=color, line_dash='dashed',
                               visible=False, legend_label=onset_label)
            if i == 0:
                p.legend.location = "top_right"

        plots.append(p)
    return plots


