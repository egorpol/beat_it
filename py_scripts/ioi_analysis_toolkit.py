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

def analyze_iois(df, onset_column):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Inter-Onset Interval (IOI) Analysis', fontsize=16)

    ax1.plot(iois, marker='.', linestyle='-', color='teal')
    ax1.set_title('IOI Sequence')
    ax1.set_xlabel('Event Number')
    ax1.set_ylabel('Interval Duration (s)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.hist(iois, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title('IOI Distribution (Histogram)')
    ax2.set_xlabel('Interval Duration (s)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Statistics ---
    stats = {'mean': np.mean(iois), 'std': np.std(iois)}
    stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
    
    ax2.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.3f}s")
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    return iois, stats

def find_optimal_grid_period(iois, min_period=0.1, max_period=1.0, num_steps=1000, top_n_to_label=5):
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
        plt.figure(figsize=(12, 6))
        plt.plot(base_values, errors, label='Quantization Error')
        plt.title('Grid Period Finding - No Local Minima Found')
        plt.show()
        return np.array([]), np.array([])

    candidate_periods = base_values[minima_indices]
    candidate_errors = errors[minima_indices]

    # Sort candidates by error to find the best ones
    sorted_indices = np.argsort(candidate_errors)
    
    # --- Plotting ---
    plt.figure(figsize=(12, 7))
    plt.plot(base_values, errors, label='Quantization Error', color='royalblue', zorder=1)
    plt.plot(candidate_periods, candidate_errors, 'o', color='red', markersize=5, label='All Local Minima', zorder=2)

    # Highlight the single BEST minimum
    best_idx = minima_indices[sorted_indices[0]]
    plt.plot(base_values[best_idx], errors[best_idx], '*', color='gold', markersize=15, markeredgecolor='black', label='Best Candidate', zorder=4)

    # Add text labels for the top N candidates
    for i in sorted_indices[:top_n_to_label]:
        idx = minima_indices[i]
        plt.text(base_values[idx] + 0.005, errors[idx], f'{base_values[idx]:.3f}s', color='black', zorder=3,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.title('Grid Period Finding by Error Minimization')
    plt.xlabel('Candidate Grid Period (s)')
    plt.ylabel('Error (Sum of Squared Residuals)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    return candidate_periods[sorted_indices], candidate_errors[sorted_indices]

def evaluate_grid_fit(iois, base_period):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Evaluation of Grid Fit for Period = {base_period:.3f}s', fontsize=16)

    ax1.plot(df_results['Original_IOI'], 'o-', label='Original IOIs', color='teal', alpha=0.8)
    ax1.plot(df_results['Reconstructed_IOI'], 'x--', label='Reconstructed Grid', color='coral', markersize=8)
    ax1.set_title('Original vs. Reconstructed IOIs'); ax1.set_xlabel('Event Number'); ax1.set_ylabel('IOI Duration (s)'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.hist(df_results['Residual'], bins=20, color='lightcoral', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='-', label='Perfect Grid Alignment')
    ax2.axvline(mean_residual_s, color='blue', linestyle='--', label=f'Mean Dev: {mean_residual_ms:+.1f}ms')
    ax2.set_title('Distribution of Residuals (Timing Deviations)'); ax2.set_xlabel('Residual Duration (s)'); ax2.set_ylabel('Frequency'); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.show()

    return df_results