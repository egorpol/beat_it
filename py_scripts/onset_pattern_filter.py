import numpy as np
from typing import List, Union, Dict, Tuple


def build_interval_graph(onset_times: List[float],
                         pattern: Union[float, List[float]],
                         tolerance: float = 0.1
                         ) -> Dict[float, List[float]]:
    """
    Build a DAG linking onset times if their interval matches pattern steps.
    (Used for looping pattern search)
    """
    edges: Dict[float, List[float]] = {t: [] for t in onset_times}
    if isinstance(pattern, (int, float)):
        pattern = [pattern]

    for i, t in enumerate(onset_times):
        for j in range(i+1, len(onset_times)):
            dt = onset_times[j] - t
            for p in pattern:
                if abs(dt - p) <= tolerance:
                    edges[t].append(onset_times[j])
                    break
    return edges


def longest_paths(edges: Dict[float, List[float]]) -> List[List[float]]:
    """
    Return all longest paths in the DAG.
    (Used for looping pattern search)
    """
    memo: Dict[float, List[float]] = {}

    def dfs(u: float) -> List[float]:
        if u in memo:
            return memo[u]
        best = [u]
        for v in edges[u]:
            path = dfs(v)
            if len(path) + 1 > len(best):
                best = [u] + path
        memo[u] = best
        return best

    all_paths: List[List[float]] = []
    max_len = 0
    for u in edges:
        p = dfs(u)
        if len(p) > max_len:
            max_len = len(p)
            all_paths = [p]
        elif len(p) == max_len:
            all_paths.append(p)
    return all_paths


def compute_fit_score(path: List[float],
                      pattern: Union[float, List[float]],
                      loop: bool) -> float:
    """
    Compute mean squared error between path intervals and pattern.
    Handles both looping and non-looping cases.
    """
    pattern_list = pattern if isinstance(pattern, list) else [pattern]
    intervals = np.diff(path)
    
    if loop:
        # For looping patterns, tile the pattern to match the interval length
        rep = np.tile(pattern_list, int(np.ceil(len(intervals) / len(pattern_list))))[:len(intervals)]
    else:
        # For fixed patterns, use the pattern directly
        rep = np.array(pattern_list)
        
    return float(np.mean((intervals - rep)**2))


def filter_onset_patterns(onset_times: List[float],
                          pattern: Union[float, List[float]],
                          tolerance: float = 0.1,
                          loop: bool = True
                          ) -> List[Tuple[np.ndarray, float]]:
    """
    Return onset-time sequences matching a pattern, sorted by fit score.

    :param onset_times: A list of timestamps.
    :param pattern: A single interval (float) or a list of intervals (list[float]).
    :param tolerance: The allowed deviation for each interval.
    :param loop: If True (default), finds the longest sequences matching a repeating pattern.
                 If False, finds all occurrences of a non-looping, fixed-length pattern.
    :returns: list of (sequence_array, fit_score)
    """
    paths: List[List[float]] = []
    
    if loop:
        # Original logic for finding the longest paths with a repeating pattern
        edges = build_interval_graph(onset_times, pattern, tolerance)
        paths = longest_paths(edges)
    else:
        # New logic for finding all occurrences of a fixed (non-looping) pattern
        pattern_list = pattern if isinstance(pattern, list) else [pattern]

        def find_sequences_recursive(start_index: int, pattern_index: int, current_path: List[float]):
            """Recursively search for the fixed sequence of intervals."""
            # If all intervals in the pattern have been matched, save the path
            if pattern_index == len(pattern_list):
                paths.append(current_path)
                return

            last_onset = current_path[-1]
            target_interval = pattern_list[pattern_index]

            # Look for the next onset that matches the current interval in the pattern
            for i in range(start_index + 1, len(onset_times)):
                dt = onset_times[i] - last_onset
                if abs(dt - target_interval) <= tolerance:
                    # If a match is found, continue searching for the next interval
                    find_sequences_recursive(i, pattern_index + 1, current_path + [onset_times[i]])

        # Start a new search from every onset time
        for i in range(len(onset_times)):
            find_sequences_recursive(i, 0, [onset_times[i]])
            
    scored: List[Tuple[np.ndarray, float]] = []
    for p in paths:
        if len(p) < 2:
            continue
        score = compute_fit_score(p, pattern, loop)
        scored.append((np.array(p), score))
        
    scored.sort(key=lambda x: x[1])
    return scored