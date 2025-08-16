# ioi_analysis_toolkit.py

"""
A toolkit for Inter-Onset Interval (IOI) analysis, including functions for 
calculating IOIs, finding optimal rhythmic grids, and evaluating the fit of a 
grid to the performance data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_iois(
    df,
    onset_column,
    *,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    layout: str | None = None,
    show_legend: bool = True,
):
    """
    Calculates Inter-Onset Intervals (IOIs) from a DataFrame and visualizes the results.

    Args:
        df (pd.DataFrame): The DataFrame containing the event data.
        onset_column (str): The name of the column with event onset times in seconds.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of the calculated IOIs.
            - dict: A dictionary of basic statistics (mean, std, cv).
    """
    if onset_column not in df.columns:
        print(f"Error: Column '{onset_column}' not found. Available columns: {list(df.columns)}")
        return np.array([]), {}
    
    onset_times = df[onset_column].dropna().values
    if len(onset_times) < 2:
        print("Error: Not enough data points to calculate IOIs.")
        return np.array([]), {}

    iois = np.diff(onset_times)
    
    # --- Visualization ---
    _layout = (layout or 'horizontal').lower()
    if _layout == 'vertical':
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=dpi, constrained_layout=True)
        ax1, ax2 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi, constrained_layout=True)
    # Removed figure-level title to avoid redundancy with printed context

    ax1.plot(iois, marker=('o' if grayscale else '.'), linestyle='-', color=('black' if grayscale else 'teal'))
    ax1.set_title('IOI Sequence', fontsize=int(12 * font_scale))
    ax1.set_xlabel('Event Number', fontsize=int(11 * font_scale))
    ax1.set_ylabel('Interval Duration (s)', fontsize=int(11 * font_scale))
    ax1.tick_params(axis='both', which='both', labelsize=int(10 * font_scale))
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.hist(iois, bins=30, color=('gray' if grayscale else 'skyblue'), edgecolor='black', alpha=0.85)
    ax2.set_title('IOI Distribution (Histogram)', fontsize=int(12 * font_scale))
    ax2.set_xlabel('Interval Duration (s)', fontsize=int(11 * font_scale))
    ax2.set_ylabel('Frequency', fontsize=int(11 * font_scale))
    ax2.tick_params(axis='both', which='both', labelsize=int(10 * font_scale))
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- Statistics ---
    stats = {'mean': np.mean(iois), 'std': np.std(iois)}
    stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
    
    ax2.axvline(stats['mean'], color=('black' if grayscale else 'red'), linestyle='--', label=f"Mean: {stats['mean']:.3f}s")
    if show_legend:
        ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    return iois, stats

def find_optimal_grid_period(
    iois,
    min_period=0.1,
    max_period=1.0,
    num_steps=1000,
    top_n_to_label=5,
    *,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    show_legend: bool = True,
):
    """
    Searches for optimal grid periods by minimizing a quantization error function.

    Args:
        iois (np.ndarray): Array of Inter-Onset Intervals.
        min_period (float): The minimum period to test.
        max_period (float): The maximum period to test.
        num_steps (int): The number of steps in the search range.
        top_n_to_label (int): The number of best candidates to label on the plot.

    Returns:
        tuple: A tuple containing two sorted numpy arrays:
            - candidate_periods (np.ndarray): Array of periods at local minima, sorted by error.
            - candidate_errors (np.ndarray): Array of error values at those minima, sorted by error.
    """
    base_values = np.linspace(min_period, max_period, num_steps)
    errors = np.array([np.sum((iois - v * np.round(iois / v))**2) for v in base_values])
    
    minima_indices, _ = find_peaks(-errors, distance=5)

    if minima_indices.size == 0:
        plt.figure(figsize=(12, 6), dpi=dpi)
        plt.plot(base_values, errors, label='Quantization Error', color=('black' if grayscale else 'C0'))
        plt.title('Grid Period Finding - No Local Minima Found', fontsize=int(14 * font_scale))
        plt.show()
        return np.array([]), np.array([])

    candidate_periods = base_values[minima_indices]
    candidate_errors = errors[minima_indices]

    # Sort candidates by error to find the best ones
    sorted_indices = np.argsort(candidate_errors)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi, constrained_layout=True)
    ax.plot(base_values, errors, label='Quantization Error', color=('black' if grayscale else 'royalblue'), zorder=1)
    # Distinctive local minima marker
    ax.plot(
        candidate_periods,
        candidate_errors,
        marker=".", linestyle='None',
        color=('black' if grayscale else 'red'),
        markersize=8, markeredgewidth=1.8,
        label='Local Minima', zorder=2,
    )

    # Highlight the single BEST minimum (yellow star with strong black edge for grayscale visibility)
    best_idx = minima_indices[sorted_indices[0]]
    ax.plot(
        base_values[best_idx],
        errors[best_idx],
        '*',
        color=('black' if grayscale else 'gold'),
        markersize=15,
        markeredgecolor='black',
        markeredgewidth=2.0,
        label='Best Candidate',
        zorder=5,
    )

    # Label only the best candidate value
    # Place label near but not overlapping the star, with an arrow for clarity
    _x = base_values[best_idx]
    _y = errors[best_idx]
    ax.annotate(
        f'{_x:.3f}s',
        xy=(_x, _y),
        xytext=(_x + 0.02 * (max(base_values)-min(base_values)), _y + 0.05 * (max(errors)-min(errors))),
        textcoords='data',
        arrowprops=dict(arrowstyle='-', color='black', lw=1.0),
        fontsize=int(10 * font_scale),
        color='black',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.2'),
        zorder=6,
    )

    ax.set_title('Grid Period Finding by Error Minimization', fontsize=int(14 * font_scale))
    ax.set_xlabel('Candidate Grid Period (s)', fontsize=int(12 * font_scale))
    ax.set_ylabel('Error (Sum of Squared Residuals)', fontsize=int(12 * font_scale))
    ax.tick_params(axis='both', which='both', labelsize=int(10 * font_scale))
    ax.grid(True, linestyle=':', alpha=0.6)
    if show_legend:
        ax.legend()
    plt.show()
    
    return candidate_periods[sorted_indices], candidate_errors[sorted_indices]

def evaluate_grid_fit(
    iois,
    base_period,
    *,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    layout: str | None = None,
    show_legend: bool = True,
):
    """
    Evaluates and visualizes how well a given grid period fits the IOI data.

    Args:
        iois (np.ndarray): Array of Inter-Onset Intervals.
        base_period (float): The grid period to evaluate.

    Returns:
        pd.DataFrame: A DataFrame containing the original IOIs, multipliers, reconstructed IOIs, and residuals.
    """
    multipliers = np.round(iois / base_period)
    reconstructed_iois = base_period * multipliers
    residuals = iois - reconstructed_iois
    df_results = pd.DataFrame({
        'Original_IOI': iois,
        'Multiplier': multipliers.astype(int),
        'Reconstructed_IOI': reconstructed_iois,
        'Residual': residuals
    })

    # --- Calculate and display residual statistics ---
    mean_residual_s = df_results['Residual'].mean()
    median_residual_s = df_results['Residual'].median()
    std_dev_s = df_results['Residual'].std()
    mean_residual_ms = mean_residual_s * 1000
    median_residual_ms = median_residual_s * 1000

    print("\n--- Timing Deviation (Residual) Analysis ---")
    print(f"Mean Residual:   {mean_residual_ms:+.2f} ms")
    print(f"Median Residual: {median_residual_ms:+.2f} ms")
    print(f"Std. Deviation:  {std_dev_s * 1000:.2f} ms (Jitter)")

    if abs(mean_residual_ms) < 5:
        print("Interpretation: The timing is very close to the grid model (low bias).")
    elif mean_residual_ms > 0:
        print("Interpretation: On average, the events are played LATE compared to the grid (dragging).")
    else:
        print("Interpretation: On average, the events are played EARLY compared to the grid (rushing).")
    print("--------------------------------------------")

    # --- Plotting ---
    _layout = (layout or 'horizontal').lower()
    if _layout == 'vertical':
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=dpi, constrained_layout=True)
        ax1, ax2 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi, constrained_layout=True)
    # Removed figure-level title to avoid redundancy with printed context

    ax1.plot(df_results['Original_IOI'], 'o-', label='Original IOIs', color=('black' if grayscale else 'teal'), alpha=0.8)
    ax1.plot(df_results['Reconstructed_IOI'], 'x--', label='Reconstructed Grid', color=('gray' if grayscale else 'coral'), markersize=8)
    ax1.set_title('Original vs. Reconstructed IOIs', fontsize=int(12 * font_scale))
    ax1.set_xlabel('Event Number', fontsize=int(11 * font_scale))
    ax1.set_ylabel('IOI Duration (s)', fontsize=int(11 * font_scale))
    ax1.tick_params(axis='both', which='both', labelsize=int(10 * font_scale))
    if show_legend:
        ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.hist(df_results['Residual'], bins=20, color=('gray' if grayscale else 'lightcoral'), edgecolor='black')
    ax2.axvline(mean_residual_s, color=('black' if grayscale else 'blue'), linestyle='--', label=f'Mean Dev: {mean_residual_ms:+.1f}ms')
    ax2.set_title('Distribution of Residuals (Timing Deviations)', fontsize=int(12 * font_scale))
    ax2.set_xlabel('Residual Duration (s)', fontsize=int(11 * font_scale))
    ax2.set_ylabel('Frequency', fontsize=int(11 * font_scale))
    ax2.tick_params(axis='both', which='both', labelsize=int(10 * font_scale))
    if show_legend:
        ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.show()

    return df_results


# ---------------- Residual Pattern Analysis Utilities ---------------- #
from typing import Dict as _Dict, Any as _Any

def residual_pattern_analysis(
    residuals_seconds: np.ndarray,
    *,
    moving_avg_window: int = 2,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    show_residuals: bool = True,
    show_autocorr: bool = True,
    show_ssm: bool = True,
    show_fft: bool = True,
    show_legend: bool = True,
    ssm_metric: str = 'l1',
    window_length: int = 11,
) -> _Dict[str, _Any]:
    """
    Produce residual timing analysis visuals and simple diagnostics.

    - residuals_seconds: 1D array of residuals in seconds
    - moving_avg_window: centered window size for smoothing (events)
    - ssm_metric: 'l1' (absolute difference) or 'cosine' (cosine similarity over windows)
    - window_length: odd length for cosine SSM windows (events)
    Returns a dict with keys: n, primary_lag, peak_period_events
    """
    import matplotlib.pyplot as _plt
    from pandas.plotting import autocorrelation_plot as _autocorr_plot
    import pandas as _pd
    from scipy.signal import find_peaks as _find_peaks
    from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq

    residuals_seconds = np.asarray(residuals_seconds, dtype=float)
    residuals_ms = residuals_seconds * 1000.0
    n = residuals_ms.size
    print(f"Analyzing {n} timing residuals...")

    title_size = int(14 * font_scale)
    label_size = int(12 * font_scale)
    tick_size = int(10 * font_scale)

    # Method 1: Residuals vs Event with smoothing
    smoothed = _pd.Series(residuals_ms).rolling(window=max(1, int(moving_avg_window)), center=True).mean().fillna(0)
    if show_residuals:
        print("\n--- Method 1: Residuals vs. Event Number ---")
        _plt.figure(figsize=(15, 5), dpi=dpi)
        _plt.plot(residuals_ms, 'o-', color=('gray' if grayscale else 'lightgray'), markersize=3, label='Raw Residuals', zorder=1)
        _plt.plot(smoothed.values, color=('black' if grayscale else 'crimson'), linewidth=2, label=f'{moving_avg_window}-Event Moving Average', zorder=2)
        _plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Perfect Grid Timing')
        _plt.title('Timing Deviations (Residuals) Over Time', fontsize=title_size)
        _plt.xlabel('Event (Note) Number', fontsize=label_size)
        _plt.ylabel('Timing Deviation (ms)\n(Early < 0 | Late > 0)', fontsize=label_size)
        _plt.tick_params(axis='both', which='both', labelsize=tick_size)
        _plt.grid(True, linestyle=':', alpha=0.6)
        if show_legend:
            _plt.legend()
        _plt.show()

    primary_lag = None
    if show_autocorr:
        # Method 2: Autocorrelation
        print("\n--- Method 2: Autocorrelation of Residuals ---")
        _plt.figure(figsize=(15, 5), dpi=dpi)
        ax_ac = _autocorr_plot(residuals_ms)
        if grayscale and hasattr(ax_ac, 'get_lines'):
            for _ln in ax_ac.get_lines():
                try:
                    _ln.set_color('black')
                except Exception:
                    pass
        _plt.title('Autocorrelation of Timing Deviations', fontsize=title_size)
        _plt.xlabel('Lag (in number of events)', fontsize=label_size)
        _plt.ylabel('Correlation Coefficient', fontsize=label_size)
        _plt.tick_params(axis='both', which='both', labelsize=tick_size)
        _plt.grid(True, linestyle=':', alpha=0.6)
        _plt.show()

        # Numeric autocorr for peak finding
        analysis_signal = smoothed.values
        ac = np.correlate(analysis_signal - analysis_signal.mean(), analysis_signal - analysis_signal.mean(), mode='full')
        ac = ac[ac.size // 2:]
        peaks, _ = _find_peaks(ac, height=np.std(ac), distance=2)
        if peaks.size > 0:
            primary_lag = int(peaks[0])
            print(f"  > RESULT: Found a significant peak at lag={primary_lag}.")
            print(f"  > INTERPRETATION: This suggests a repeating pattern every {primary_lag} events.")
        else:
            print("  > RESULT: No significant autocorrelation peaks found.")
            print("  > INTERPRETATION: The timing deviations do not have a strong, regularly repeating pattern.")

    peak_period = None
    if show_ssm:
        # Method 3: Self-similarity matrix
        print("\n--- Method 3: Self-Similarity Matrix ---")
        metric = (ssm_metric or 'l1').lower()
        _plt.figure(figsize=(9, 9), dpi=dpi)
        if metric == 'cosine':
            # Ensure odd window length
            if window_length < 1:
                window_length = 1
            if window_length % 2 == 0:
                window_length += 1
            half = window_length // 2
            # Build windowed matrix (n, window_length) with edge padding
            n = residuals_ms.size
            W = np.empty((n, window_length), dtype=float)
            for i in range(n):
                start = i - half
                end = i + half + 1
                # Clamp indices
                left_pad = max(0, -start)
                right_pad = max(0, end - n)
                valid_start = max(0, start)
                valid_end = min(n, end)
                segment = residuals_ms[valid_start:valid_end]
                if left_pad > 0:
                    segment = np.concatenate([np.full(left_pad, residuals_ms[0]), segment])
                if right_pad > 0:
                    segment = np.concatenate([segment, np.full(right_pad, residuals_ms[-1])])
                W[i, :] = segment[:window_length]
            # Z-score windows to emphasize shape over magnitude
            W_mean = W.mean(axis=1, keepdims=True)
            W_std = W.std(axis=1, keepdims=True)
            Wz = np.where(W_std > 0, (W - W_mean) / W_std, 0.0)
            # Cosine similarity matrix
            norms = np.linalg.norm(Wz, axis=1, keepdims=True)
            denom = (norms @ norms.T) + 1e-12
            S = (Wz @ Wz.T) / denom
            # Map from [-1, 1] to [0, 1] for display
            S_vis = (S + 1.0) / 2.0
            _plt.imshow(S_vis, cmap=('Greys' if grayscale else 'magma_r'), interpolation='nearest', origin='lower', vmin=0.0, vmax=1.0)
            _plt.title(f'Self-Similarity (Cosine, window={window_length})', fontsize=title_size)
            _plt.xlabel('Event Number', fontsize=label_size)
            _plt.ylabel('Event Number', fontsize=label_size)
            _plt.tick_params(axis='both', which='both', labelsize=tick_size)
            cbar = _plt.colorbar()
            cbar.set_label('Cosine Similarity (0–1)')
        else:
            # L1 absolute difference (ms)
            residuals_col = residuals_ms[:, np.newaxis]
            residuals_row = residuals_ms[np.newaxis, :]
            ssm = np.abs(residuals_col - residuals_row)
            _plt.imshow(ssm, cmap=('Greys' if grayscale else 'magma_r'), interpolation='nearest', origin='lower')
            _plt.title('Self-Similarity Matrix of Residuals', fontsize=title_size)
            _plt.xlabel('Event Number', fontsize=label_size)
            _plt.ylabel('Event Number', fontsize=label_size)
            _plt.tick_params(axis='both', which='both', labelsize=tick_size)
            cbar = _plt.colorbar()
            cbar.set_label('Absolute Difference in Timing (ms)')
        _plt.show()

    if show_fft:
        # Method 4: FFT analysis
        print("\n--- Method 4: FFT of Smoothed Residuals ---")
        analysis_signal = smoothed.values
        N = analysis_signal.size
        amps = np.abs(_rfft(analysis_signal))
        xf = _rfftfreq(N, 1.0)
        if xf.size > 1:
            peaks, _ = _find_peaks(amps, height=np.std(amps))
            if peaks.size > 0:
                strongest_idx = int(peaks[np.argmax(amps[peaks])])
                peak_freq = float(xf[strongest_idx])
                if peak_freq > 0:
                    peak_period = 1.0 / peak_freq
                print(f"  > RESULT: Dominant frequency at {peak_freq:.3f} cycles/event -> period {peak_period:.2f} events.")
                _plt.figure(figsize=(15, 5), dpi=dpi)
                _plt.plot(xf[1:], amps[1:], color=('black' if grayscale else 'navy'))
                _plt.plot(peak_freq, amps[strongest_idx], 'x', color=('black' if grayscale else 'red'), markersize=12,
                         label=(f'Strongest Cycle: {peak_period:.2f} events' if peak_period else 'Strongest Cycle'))
                _plt.title('Frequency Spectrum of Smoothed Residuals (FFT)', fontsize=title_size)
                _plt.xlabel('Frequency (cycles per event)', fontsize=label_size)
                _plt.ylabel('Amplitude', fontsize=label_size)
                _plt.tick_params(axis='both', which='both', labelsize=tick_size)
                _plt.grid(True, which='both', linestyle=':', alpha=0.6)
                if show_legend:
                    _plt.legend()
                _plt.show()
            else:
                print("  > RESULT: No dominant frequency component found.")
        else:
            print("  > RESULT: Not enough data for FFT analysis.")

    # Summary
    print("\n[C] Overall Conclusion")
    if primary_lag is not None and peak_period is not None and abs(primary_lag - peak_period) < 0.5:
        print(f"  Consistent repeating pattern ~{primary_lag} events; FFT corroborates (~{peak_period:.2f}).")
        print("  CONCLUSION: Strong evidence of a repeating rhythmic feel.")
    elif primary_lag is not None:
        print(f"  Autocorrelation suggests a repeating pattern ~{primary_lag} events.")
        print("  CONCLUSION: Moderate evidence of a repeating pattern.")
    elif peak_period is not None:
        print(f"  FFT suggests a cycle of ~{peak_period:.2f} events without autocorrelation support.")
        print("  CONCLUSION: Weak/complex periodic pattern suspected.")
    else:
        print("  No strong periodicity found; residuals may be random jitter or drift.")

    return {"n": int(n), "primary_lag": (None if primary_lag is None else int(primary_lag)),
            "peak_period_events": (None if peak_period is None else float(peak_period))}


def residual_sensitivity_and_summary(
    residuals_seconds: np.ndarray,
    *,
    window_sizes_to_test: list[int] | None = None,
    main_moving_avg_window: int = 2,
    grayscale: bool = True,
    font_scale: float = 1.8,
    dpi: int | None = 300,
    show_legend: bool = True,
) -> _Dict[str, _Any]:
    """
    Replicates the notebook's Advanced Residual Pattern Analysis:
    - Sensitivity across multiple smoothing windows (prints dominant lags)
    - Automated analysis with autocorrelation and FFT (prints findings)
    - Summary plot showing residuals, smoothed trend, and FFT cycle markers

    Returns dict with keys: n, primary_lag, peak_period_events.
    """
    import numpy as _np
    import pandas as _pd
    import matplotlib.pyplot as _plt
    from scipy.signal import find_peaks as _find_peaks
    from scipy.fft import rfft as _rfft, rfftfreq as _rfftfreq

    residuals_seconds = _np.asarray(residuals_seconds, dtype=float)
    residuals_ms = residuals_seconds * 1000.0
    n = residuals_ms.size
    print(f"Analyzing {n} timing residuals...")

    title_size = int(16 * font_scale)
    label_size = int(12 * font_scale)
    tick_size = int(10 * font_scale)

    # Sensitivity Analysis
    print("\n\n" + "="*60)
    print("--- Method 1: Sensitivity Analysis for Smoothing Window ---")
    print("="*60)
    print("Analyzing which patterns are strongest at different levels of smoothing.")
    if window_sizes_to_test is None:
        window_sizes_to_test = [2, 3, 4, 8, 12, 16]
    for window in window_sizes_to_test:
        if n < window:
            continue
        smoothed = _pd.Series(residuals_ms).rolling(window=window, center=True).mean().fillna(0).values
        ac = _np.correlate(smoothed - smoothed.mean(), smoothed - smoothed.mean(), mode='full')
        ac = ac[ac.size // 2:]
        peaks, _ = _find_peaks(ac, height=_np.std(ac) * 1.5, distance=2)
        if peaks.size > 0:
            print(f"  - For window size {window:<2}: Strongest pattern repeats every {int(peaks[0])} events.")
        else:
            print(f"  - For window size {window:<2}: No dominant repeating pattern found.")
    print("\n  > INTERPRETATION: Look for a lag number that appears consistently across")
    print("    different window sizes. A consistent result indicates a robust pattern.")

    # Automated Analysis
    print("\n\n" + "="*60)
    print(f"--- Method 2: Detailed Analysis for Window = {main_moving_avg_window} ---")
    print("="*60)
    analysis_signal = _pd.Series(residuals_ms).rolling(window=main_moving_avg_window, center=True).mean().fillna(0).values
    primary_lag = None
    ac = _np.correlate(analysis_signal - analysis_signal.mean(), analysis_signal - analysis_signal.mean(), mode='full')
    ac = ac[ac.size // 2:]
    peaks, _ = _find_peaks(ac, height=_np.std(ac), distance=2)
    if peaks.size > 0:
        primary_lag = int(peaks[0])
        print(f"[Autocorrelation] Found a repeating motif every {primary_lag} events.")
    else:
        print("[Autocorrelation] No significant repeating motif found.")

    peak_period = None
    N = analysis_signal.size
    yf = _rfft(analysis_signal)
    xf = _rfftfreq(N, 1)
    if xf.size > 1:
        amps = _np.abs(yf)
        fft_peaks, _ = _find_peaks(amps[1:], height=_np.std(amps[1:]), distance=2)
        if fft_peaks.size > 0:
            strongest_peak_idx = int(fft_peaks[_np.argmax(amps[1:][fft_peaks])]) + 1
            peak_freq = float(xf[strongest_peak_idx])
            if peak_freq > 0:
                peak_period = 1.0 / peak_freq
            print(f"[FFT]             Found a cyclical wave every {peak_period:.2f} events.")
        else:
            print("[FFT]             No dominant cyclical wave found.")
    else:
        print("[FFT]             Not enough data for FFT analysis.")

    # Summary Visualization
    print("\n\n" + "="*60)
    print("--- Method 3: Summary Visualization with Detected Cycles ---")
    print("="*60)
    _plt.figure(figsize=(18, 7), dpi=dpi)
    _plt.plot(residuals_ms, 'o-', color=('gray' if grayscale else 'silver'), markersize=4, alpha=0.7, label='Raw Residuals', zorder=1)
    _plt.plot(analysis_signal, color=('black' if grayscale else 'navy'), linewidth=2.5, label=f'{main_moving_avg_window}-Event Smoothed Trend', zorder=2)
    if peak_period is not None and peak_period > 0:
        plot_period = int(round(peak_period))
        for i in range(plot_period, n, plot_period):
            _plt.axvline(x=i, color=('black' if grayscale else 'red'), linestyle=':', linewidth=1.5, alpha=0.8, zorder=3)
        _plt.axvline(x=-1, color=('black' if grayscale else 'red'), linestyle=':', linewidth=1.5, label=f'FFT Cycle ({peak_period:.1f} events)')
    _plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label='Perfect Grid Timing', zorder=2)
    _plt.title('Timing Deviations with Detected Rhythmic Cycles', fontsize=title_size)
    _plt.xlabel('Event (Note) Number', fontsize=label_size)
    _plt.ylabel('Timing Deviation (ms)\n(Early < 0 | Late > 0)', fontsize=label_size)
    _plt.grid(True, which='both', linestyle=':', alpha=0.6)
    _plt.tick_params(axis='both', which='both', labelsize=tick_size)
    if show_legend:
        _plt.legend()
    _plt.xlim(0, n-1)
    _plt.show()

    return {"n": int(n), "primary_lag": (None if primary_lag is None else int(primary_lag)),
            "peak_period_events": (None if peak_period is None else float(peak_period))}