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

def _time_str_to_seconds(time_str: str) -> float:
    """
    Converts a flexible time string to seconds.
    Supports formats like: 'SS.fff', 'MM:SS.fff', 'HH:MM:SS.fff'.
    Examples: '10.5', '01:30.250', '00:01:30.5'.
    """
    time_str = time_str.strip()
    parts = time_str.split(':')
    
    if len(parts) == 1:  # Only seconds (e.g., '10.5')
        return float(parts[0])
    
    if len(parts) == 2:  # M:S (e.g., '01:30.5')
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
        
    if len(parts) == 3:  # H:M:S (e.g., '00:01:30.5')
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        
    raise ValueError(f"Invalid time format: '{time_str}'. Use SS.fff, MM:SS.fff, or HH:MM:SS.fff.")

def _parse_time_slice(time_slice: str) -> tuple[float, float]:
    """Parses a 'start-end' string into offset and duration."""
    try:
        start_str, end_str = time_slice.split('-')
        start_s = _time_str_to_seconds(start_str)
        end_s = _time_str_to_seconds(end_str)
        if end_s <= start_s:
            raise ValueError("End time must be after start time.")
        return start_s, end_s - start_s
    except ValueError as e:
        raise ValueError(f"Invalid time slice format '{time_slice}'. Use formats like '10.5-20.75' or '00:10.5-00:20.75'.") from e

def load_audio(
    path: str,
    sr: Optional[int] = None,
    time_slice: Optional[str] = None
) -> tuple[np.ndarray, int]:
    """
    Load an audio file or a slice of it.

    Args:
        path (str): Path to the audio file.
        sr (int, optional): Target sample rate. Defaults to None.
        time_slice (str, optional): Slice to load, e.g., "10-20" or "00:10-00:20".
                                    Defaults to None (loads full file).

    Returns:
        tuple[np.ndarray, int]: Audio time series and sample rate.
    """
    offset, duration = None, None
    load_msg = f"Loaded '{path}'"

    if time_slice:
        offset, duration = _parse_time_slice(time_slice)
        load_msg = f"Loaded slice {offset:.2f}s-{offset+duration:.2f}s from '{path}'"

    y, sr = librosa.load(path, sr=sr, offset=offset, duration=duration)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"{load_msg} -> {total_duration:.2f}s @ {sr}Hz")
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
    # Updated type hint to show it can be a list, array, or dict
    onset_times: Optional[Union[np.ndarray, list, dict]] = None,
    features_to_plot: List[str] = None,
    backend: str = 'matplotlib'
) -> None:
    valid = list(features.keys())
    if features_to_plot is None:
        features_to_plot = ['waveform', 'stft_db', 'spectral_flux']
    features_to_plot = [f for f in features_to_plot if f in valid]

    if backend.lower() == 'matplotlib':
        _plot_matplotlib(features, sr, hop_length, onset_times, features_to_plot)
    elif backend.lower() == 'bokeh':
        from bokeh.io import show
        from bokeh.layouts import column
        plots = _plot_bokeh(features, sr, hop_length, onset_times, features_to_plot)
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
    features_to_plot: List[str]
) -> None:
    n = len(features_to_plot)
    # Turn on constrained_layout here:
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n),
                             sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    duration = librosa.get_duration(y=features['waveform'], sr=sr)

    # Build one set of legend handles up front
    global_legend_handles = []
    if isinstance(onset_times, dict):
        for label, data_dict in onset_times.items():
            color = data_dict.get('color', 'r')
            proxy = Line2D([0], [0], linestyle='--',
                           color=color, label=label)
            global_legend_handles.append(proxy)

    # Plot each feature + its onset lines
    for ax, name in zip(axes, features_to_plot):
        data = features[name]

        # 1D vs 2D plotting
        if data.ndim == 1:
            if name == 'waveform':
                arr = data[::max(1, len(data)//10000)]
                times = np.linspace(0, duration, len(arr))
            else:
                arr = data
                times = librosa.times_like(data, sr=sr,
                                           hop_length=hop_length)
            ax.plot(times, arr)
            ax.set_ylabel(FEATURE_LABELS.get(name, name))
        else:
            y_axis = 'mel' if 'melspectrogram' in name else 'linear'
            librosa.display.specshow(
                data, sr=sr, hop_length=hop_length,
                x_axis='time', y_axis=y_axis, ax=ax
            )
            ax.set_ylabel('Freq' + (' (Hz)' if y_axis=='linear'
                                    else ' (Mel)'))

        # Onset lines
        if onset_times is not None:
            ymin, ymax = ax.get_ylim()
            if isinstance(onset_times, (list, np.ndarray)):
                ax.vlines(onset_times, ymin, ymax,
                          color='r', linestyle='--')
            else:
                for onset_data in onset_times.values():
                    ax.vlines(onset_data['times'], ymin, ymax,
                              color=onset_data.get('color', 'r'),
                              linestyle='--')

    # Final formatting
    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Time (s)')

    # One global legend in upper‐right, inside the figure padding
    if global_legend_handles:
        fig.legend(
            handles=global_legend_handles,
            loc='upper right',
            borderaxespad=1.0
        )

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
    features_to_plot: List[str]
) -> List[Any]:

    plots: List[Any] = []
    duration = librosa.get_duration(y=features['waveform'], sr=sr)
    tools = 'pan,wheel_zoom,box_zoom,reset,save'

    for i, name in enumerate(features_to_plot):

        # --- one shared x-range ---
        if i == 0:
            p = figure(width=1400, height=400,
                       x_axis_label='Time (s)', tools=tools)
        else:
            p = figure(width=1400, height=400,
                       x_axis_label='Time (s)', tools=tools,
                       x_range=plots[0].x_range)   # shared
        label = FEATURE_LABELS.get(name, name.replace('_', ' ').title())
        p.title.text = label

        # --- draw waveform / spectrum ---
        data = features[name]
        if data.ndim == 1:
            arr = data[::max(1, len(data)//5000)] if name == 'waveform' else data
            times = (np.linspace(0, duration, len(arr)) if name == 'waveform'
                     else librosa.times_like(data, sr=sr, hop_length=hop_length))
            p.line(times, arr, line_width=1)
            p.yaxis.axis_label = label
        else:
            mapper = LinearColorMapper(palette='Viridis256',
                                       low=np.min(data), high=np.max(data))
            if 'melspectrogram' in name:
                y0, dy = 0, data.shape[0]
                p.y_range = Range1d(y0, dy)
                p.yaxis.axis_label = 'Mel bins'
            else:
                y0, dy = 0, sr/2
                p.y_range = Range1d(y0, dy)
                p.yaxis.axis_label = 'Frequency (Hz)'
            p.image(image=[data], x=0, y=y0, dw=duration, dh=dy,
                    color_mapper=mapper)
            p.add_layout(ColorBar(color_mapper=mapper, title='dB'), 'right')

        # --- onset markers & legend (first plot only) ---
        if onset_times is not None:
            if isinstance(onset_times, (list, np.ndarray)):
                # spans
                for t in onset_times:
                    p.add_layout(Span(location=t, dimension='height',
                                      line_color='red', line_dash='dashed'))
                if i == 0:  # dummy glyph for legend
                    dummy = p.line([duration+1], [0], line_color='red',
                                   line_dash='dashed', visible=False,
                                   legend_label='Onsets')

            elif isinstance(onset_times, dict):
                for onset_label, onset_data in onset_times.items():
                    color = onset_data.get('color', 'red')
                    for t in onset_data['times']:
                        p.add_layout(Span(location=t, dimension='height',
                                          line_color=color,
                                          line_dash='dashed'))
                    if i == 0:  # dummy glyph
                        p.line([duration+1], [0], line_color=color,
                               line_dash='dashed', visible=False,
                               legend_label=onset_label)

            if i == 0:          # only first figure gets the legend
                p.legend.location = "top_right"
                # p.legend.click_policy = "hide"

        plots.append(p)

    return plots

