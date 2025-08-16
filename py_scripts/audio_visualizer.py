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
    grayscale: bool = True,
    grayscale_cmap: str = 'gray',
    font_family: Optional[str] = None,
    font_scale: float = 1.8,
    dpi: Optional[int] = 300,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    show_legend: bool = True,
    show_slice_markers: bool = False
) -> None:
    # Derive t_start automatically if not provided
    if t_start is None:
        t_start = parse_time_range(time_range)[0] if time_range else 0.0

    # Default feature selection if not provided
    if not features_to_plot:
        try:
            features_to_plot = list(features.keys())
        except Exception:
            features_to_plot = []

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
            show_legend=show_legend,
            show_slice_markers=show_slice_markers,
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
            show_legend=show_legend,
            show_slice_markers=show_slice_markers,
        )
        show(column(*plots))
    else:
        raise ValueError(f"Unknown backend: {backend}")



def _plot_matplotlib(
    features: Dict[str, Any],
    sr: int,
    hop_length: int,
    onset_times: Optional[Union[np.ndarray, list, dict]],
    features_to_plot: List[str],
    t_start: float,
    *,
    grayscale: bool = True,
    grayscale_cmap: str = 'gray',
    font_family: Optional[str] = None,
    font_scale: float = 1.8,
    dpi: Optional[int] = 300,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    show_legend: bool = True,
    show_slice_markers: bool = False
) -> None:
    n = len(features_to_plot)
    use_figsize = figsize if figsize is not None else (14, 4*n)
    fig, axes = plt.subplots(n, 1, figsize=use_figsize, sharex=True, constrained_layout=True, dpi=dpi)
    if n == 1:
        axes = [axes]

    seg_duration = librosa.get_duration(y=features['waveform'], sr=sr)
    seg_end = t_start + seg_duration

    global_legend_handles = []
    if isinstance(onset_times, dict):
        for label, data_dict in onset_times.items():
            style = (data_dict.get('style') or {})
            legend_color = 'black' if grayscale else data_dict.get('color', 'r')
            linestyle = style.get('line_style', '--')
            line_width = float(style.get('line_width', 1.0))
            proxy = Line2D([0], [0], linestyle=linestyle, color=legend_color, linewidth=line_width, label=label)
            dash_pattern = style.get('dash_pattern')
            if dash_pattern and hasattr(proxy, 'set_dashes'):
                proxy.set_dashes(dash_pattern)
            global_legend_handles.append(proxy)

    # Font settings per-axis
    base_fontsize = plt.rcParams.get('font.size', 10.0)
    scaled = base_fontsize * max(0.1, float(font_scale))
    label_size = scaled
    tick_size = scaled * 0.9

    for ax, name in zip(axes, features_to_plot):
        if font_family:
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontfamily(font_family)
        ax.tick_params(axis='both', which='both', labelsize=tick_size)
        data = features[name]
        if data.ndim == 1:
            if name == 'waveform':
                arr = data[::max(1, len(data)//10000)]
                times = t_start + np.linspace(0, seg_duration, len(arr))
            else:
                times = t_start + librosa.times_like(data, sr=sr, hop_length=hop_length)
                arr = data
            line_color = 'black' if grayscale else None
            ax.plot(times, arr, color=line_color)
            ax.set_ylabel(FEATURE_LABELS.get(name, name), fontsize=label_size, fontfamily=font_family)
        else:
            # Time-aligned spectrogram using explicit x_coords
            frame_times = t_start + librosa.times_like(data, sr=sr, hop_length=hop_length)
            y_axis = 'mel' if 'melspectrogram' in name else 'linear'
            librosa.display.specshow(
                data, sr=sr, hop_length=hop_length,
                x_axis='time', y_axis=y_axis, ax=ax, x_coords=frame_times,
                cmap=(grayscale_cmap if grayscale else None)
            )
            ax.set_ylabel('Freq' + (' (Hz)' if y_axis == 'linear' else ' (Mel)'), fontsize=label_size, fontfamily=font_family)

        if onset_times is not None:
            ymin, ymax = ax.get_ylim()
            full_y0, full_y1 = ymin, ymax
            if isinstance(onset_times, (list, np.ndarray)):
                # Single-series style (global)
                onset_color = 'black' if grayscale else 'r'
                linestyle = '--'
                line_width = 1.0
                ax.vlines(onset_times, full_y0, full_y1, color=onset_color, linestyle=linestyle, linewidth=line_width)
            else:
                for onset_label, onset_data in onset_times.items():
                    style = (onset_data.get('style') or {})
                    color = 'black' if grayscale else onset_data.get('color', 'r')
                    linestyle = style.get('line_style', '--')
                    line_width = float(style.get('line_width', 1.0))
                    tick_h_frac = style.get('tick_height')
                    tick_pos = (style.get('tick_position') or 'full').lower()

                    # Compute y span
                    if isinstance(tick_h_frac, (int, float)) and tick_h_frac > 0:
                        span = (ymax - ymin) * min(1.0, max(0.0, float(tick_h_frac)))
                        if tick_pos == 'top':
                            y0, y1 = ymax - span, ymax
                        elif tick_pos == 'bottom':
                            y0, y1 = ymin, ymin + span
                        elif tick_pos == 'middle':
                            mid = (ymax + ymin) / 2.0
                            y0, y1 = mid - span/2.0, mid + span/2.0
                        else:
                            y0, y1 = full_y0, full_y1
                    else:
                        y0, y1 = full_y0, full_y1

                    ax.vlines(onset_data['times'], y0, y1, color=color, linestyle=linestyle, linewidth=line_width)

                    # Optional marker support
                    marker = style.get('marker')
                    if marker:
                        msize = float(style.get('marker_size', 6.0))
                        if tick_pos == 'bottom':
                            y_mark = y0
                        elif tick_pos == 'middle':
                            y_mark = (y0 + y1) / 2.0
                        else:
                            y_mark = y1
                        ax.scatter(onset_data['times'], np.full_like(np.array(onset_data['times']), y_mark),
                                   marker=marker, s=msize, c=color)

        # Ensure the view spans exactly the requested slice
        ax.set_xlim(t_start, seg_end)

        # Optional slice boundary markers for debugging
        if show_slice_markers:
            ax.axvline(t_start, color='green', linestyle=':', linewidth=1.0)
            ax.axvline(seg_end, color='green', linestyle=':', linewidth=1.0)

    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Time (s)', fontsize=label_size, fontfamily=font_family)

    if show_legend and global_legend_handles:
        fig.legend(handles=global_legend_handles, loc='upper right', borderaxespad=1.0)

    # Apply figure-level font family if requested
    if font_family:
        fig.canvas.get_renderer()
        for text in fig.texts:
            text.set_fontfamily(font_family)

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=(dpi if dpi is not None else fig.dpi), bbox_inches='tight', facecolor='white')

    plt.show()



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
    t_start: float,
    *,
    grayscale: bool = True,
    font_scale: float = 1.8,
    show_legend: bool = True,
    show_slice_markers: bool = False
) -> List[Any]:
    plots: List[Any] = []
    seg_duration = librosa.get_duration(y=features['waveform'], sr=sr)
    seg_end = t_start + seg_duration
    tools = 'pan,wheel_zoom,box_zoom,reset,save'
    base_pt = 12
    title_pt = int(base_pt * max(0.1, float(font_scale)) * 1.1)
    label_pt = int(base_pt * max(0.1, float(font_scale)))
    tick_pt = int(base_pt * max(0.1, float(font_scale)) * 0.9)

    for i, name in enumerate(features_to_plot):
        if i == 0:
            p = figure(width=1400, height=400, x_axis_label='Time (s)', tools=tools)
            p.x_range = Range1d(t_start, seg_end)
        else:
            p = figure(width=1400, height=400, x_axis_label='Time (s)', tools=tools,
                       x_range=plots[0].x_range)
        label = FEATURE_LABELS.get(name, name.replace('_', ' ').title())
        p.title.text = label
        p.title.text_font_size = f"{title_pt}pt"
        p.xaxis.axis_label_text_font_size = f"{label_pt}pt"
        p.yaxis.axis_label_text_font_size = f"{label_pt}pt"
        p.xaxis.major_label_text_font_size = f"{tick_pt}pt"
        p.yaxis.major_label_text_font_size = f"{tick_pt}pt"

        data = features[name]
        if data.ndim == 1:
            arr = data[::max(1, len(data)//5000)] if name == 'waveform' else data
            times = (t_start + np.linspace(0, seg_duration, len(arr))
                     if name == 'waveform'
                     else t_start + librosa.times_like(data, sr=sr, hop_length=hop_length))
            p.line(times, arr, line_width=1, line_color=('black' if grayscale else 'navy'))
            p.yaxis.axis_label = label
        else:
            mapper = LinearColorMapper(palette=('Greys256' if grayscale else 'Viridis256'), low=np.min(data), high=np.max(data))
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
                    p.add_layout(Span(location=t, dimension='height', line_color=('black' if grayscale else 'red'), line_dash='dashed'))
                if show_legend and i == 0:
                    p.line([t_start + seg_duration + 1], [0], line_color=('black' if grayscale else 'red'), line_dash='dashed',
                           visible=False, legend_label='Onsets')
            elif isinstance(onset_times, dict):
                for onset_label, onset_data in onset_times.items():
                    style = (onset_data.get('style') or {})
                    color = 'black' if grayscale else onset_data.get('color', 'red')
                    line_dash = style.get('line_style', 'dashed')
                    line_width = float(style.get('line_width', 1.0))
                    tick_h_frac = style.get('tick_height')
                    tick_pos = (style.get('tick_position') or 'full').lower()

                    # Represent as full-height Spans or short segments (ticks)
                    if isinstance(tick_h_frac, (int, float)) and tick_h_frac > 0:
                        # Draw short ticks near the top/bottom/middle using segments
                        # Determine y0/y1 based on tick position
                        if 'melspectrogram' in name:
                            y_min, y_max = p.y_range.start, p.y_range.end
                        else:
                            y_min, y_max = p.y_range.start, p.y_range.end
                        span = (y_max - y_min) * min(1.0, max(0.0, float(tick_h_frac)))
                        if tick_pos == 'top':
                            y0, y1 = y_max - span, y_max
                        elif tick_pos == 'bottom':
                            y0, y1 = y_min, y_min + span
                        elif tick_pos == 'middle':
                            mid = (y_max + y_min) / 2.0
                            y0, y1 = mid - span/2.0, mid + span/2.0
                        else:
                            y0, y1 = y_min, y_max

                        # Draw ticks
                        for t in onset_data['times']:
                            p.segment(x0=t, y0=y0, x1=t, y1=y1, line_color=color, line_dash=line_dash, line_width=line_width)
                    else:
                        # Full-height spans
                        for t in onset_data['times']:
                            p.add_layout(Span(location=t, dimension='height', line_color=color, line_dash=line_dash, line_width=line_width))
                    if show_legend and i == 0:
                        # Legend proxy with proper dash/width in grayscale or color
                        p.line([t_start + seg_duration + 1], [0], line_color=color, line_dash=line_dash, line_width=line_width,
                               visible=False, legend_label=onset_label)
            if show_legend and i == 0:
                p.legend.location = "top_right"

        # Optional slice boundary markers for debugging
        if show_slice_markers:
            from bokeh.models import Span as BokehSpan
            p.add_layout(BokehSpan(location=t_start, dimension='height', line_color='green', line_dash='dotted', line_width=1))
            p.add_layout(BokehSpan(location=seg_end, dimension='height', line_color='green', line_dash='dotted', line_width=1))

        plots.append(p)
    return plots



from typing import Dict, List

def visualize_multi_onset_differences(
    cue_lists: Dict[str, List[float]],
    ground_truth_key: str,
    *,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    show_legend: bool = True,
    layout: str = 'horizontal',
) -> None:
    """
    Compare multiple detected onset lists against a single ground truth.
    Grayscale-friendly rendering for print-safe visuals.
    """
    if ground_truth_key not in cue_lists:
        raise ValueError(f"The ground_truth_key '{ground_truth_key}' was not found in cue_lists.")

    truth_times = np.array(cue_lists[ground_truth_key], dtype=float)
    detected_keys = [key for key in cue_lists if key != ground_truth_key]
    if not detected_keys:
        print("No other lists to compare against the ground truth.")
        return

    n_rows = len(detected_keys)
    layout = (layout or 'horizontal').lower()
    if layout == 'vertical':
        total_rows = n_rows * 2
        # Two stacked plots per series, one column
        fig, axes = plt.subplots(total_rows, 1, figsize=(14, 5 * total_rows), squeeze=False,
                                 constrained_layout=True, dpi=dpi)
    else:
        # Default: two columns per series (time error | histogram)
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows), squeeze=False,
                                 constrained_layout=True, dpi=dpi)
    title_size = int(16 * font_scale)
    label_size = int(12 * font_scale)
    tick_size = int(10 * font_scale)
    fig.suptitle(f'Onset Detection Comparison against "{ground_truth_key}"', fontsize=title_size)

    for i, key in enumerate(detected_keys):
        detected_times = np.array(cue_lists[key], dtype=float)
        min_len = min(len(truth_times), len(detected_times))
        truth_sliced = truth_times[:min_len]
        detected_sliced = detected_times[:min_len]
        error_ms = (detected_sliced - truth_sliced) * 1000.0

        # Plot 1: Timing Error vs truth time
        if layout == 'vertical':
            ax1 = axes[2*i, 0]
        else:
            ax1 = axes[i, 0]
        if grayscale:
            ax1.stem(truth_sliced, error_ms, linefmt='k-', markerfmt='ko', basefmt='k-')
            zero_color = 'k'
        else:
            ax1.stem(truth_sliced, error_ms)
            zero_color = 'r'
        ax1.axhline(0, color=zero_color, linestyle='--')
        ax1.set_title(f'Timing Error: {key}', fontsize=int(14 * font_scale))
        ax1.set_xlabel(f'Time (s) from "{ground_truth_key}"', fontsize=label_size)
        ax1.set_ylabel('Error (ms)', fontsize=label_size)
        ax1.tick_params(axis='both', which='both', labelsize=tick_size)
        ax1.grid(True, linestyle=':')

        # Plot 2: Error histogram with median line
        if layout == 'vertical':
            ax2 = axes[2*i + 1, 0]
        else:
            ax2 = axes[i, 1]
        if grayscale:
            ax2.hist(error_ms, bins=40, alpha=0.8, edgecolor='black', color='gray')
            med_color = 'k'
        else:
            ax2.hist(error_ms, bins=40, alpha=0.8, edgecolor='black')
            med_color = 'r'
        median_error = np.median(error_ms)
        ax2.axvline(median_error, color=med_color, linestyle='--', label=f'Median Error: {median_error:.2f} ms')
        ax2.set_title(f'Error Distribution: {key}', fontsize=int(14 * font_scale))
        ax2.set_xlabel('Error (ms)', fontsize=label_size)
        ax2.set_ylabel('Count', fontsize=label_size)
        ax2.tick_params(axis='both', which='both', labelsize=tick_size)
        if show_legend:
            ax2.legend()
        ax2.grid(True, linestyle=':')

    plt.show()


def plot_multi_ioi_comparison(
    cue_lists: Dict[str, List[float]],
    *,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    show_legend: bool = True,
    show_avg_line: bool = True,
) -> None:
    """
    Plot Inter-Onset Intervals (IOI) for multiple onset lists.
    - cue_lists: mapping label -> list of onset times in seconds
    - show_avg_line: draw the horizontal average IOI line for each series
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    labels = list(cue_lists.keys())
    title_size = int(18 * font_scale)
    label_size = int(12 * font_scale)
    tick_size = int(10 * font_scale)

    fig, ax = _plt.subplots(figsize=(18, 8), constrained_layout=True, dpi=dpi)

    if grayscale:
        dash_styles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1))]
        markers = ['o', 's', '^', 'x', 'D']
    else:
        cmap = _plt.cm.tab10(_np.linspace(0, 1, max(1, len(labels))))

    for i, (label, times) in enumerate(cue_lists.items()):
        times_arr = _np.array(times, dtype=float)
        if times_arr.size < 2:
            print(f"Warning: List '{label}' has fewer than 2 onsets, cannot calculate IOI. Skipping.")
            continue

        ioi = _np.diff(times_arr)
        avg_ioi = float(_np.mean(ioi))
        legend_label = f"{label} (Avg IOI: {avg_ioi:.3f}s)"

        if grayscale:
            ax.plot(
                ioi,
                marker=markers[i % len(markers)],
                linestyle=dash_styles[i % len(dash_styles)],
                markersize=4,
                color='black',
                alpha=0.85,
                label=legend_label,
            )
            if show_avg_line:
                ax.axhline(y=avg_ioi, color='black', linestyle='--', alpha=0.9)
        else:
            color = cmap[i % len(labels)] if len(labels) > 0 else 'C0'
            ax.plot(ioi, marker='o', linestyle='-', markersize=4, label=legend_label, color=color, alpha=0.7)
            if show_avg_line:
                ax.axhline(y=avg_ioi, color=color, linestyle='--', alpha=0.9)

    ax.set_title('Inter-Onset Interval (IOI) Comparison', fontsize=title_size)
    ax.set_xlabel('Onset Event Number', fontsize=label_size)
    ax.set_ylabel('Time Between Onsets (s)', fontsize=label_size)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='both', labelsize=tick_size)

    if show_legend:
        ax.legend()

    _plt.show()
