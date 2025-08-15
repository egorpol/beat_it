import numpy as np
import random
from typing import Union, List, Optional

def generate_onset_pattern(
    pattern: Union[float, List[float]],
    start_time: float = 0.0,
    num_events: int = 10,
    tolerance: Optional[float] = None
) -> np.ndarray:
    """
    Generates a list of onset timestamps based on a looping pattern.

    Args:
        pattern (Union[float, List[float]]): 
            The Inter-Onset Interval (IOI) pattern.
            - If float: A constant IOI that is repeated (e.g., 2.23 for a steady beat).
            - If List[float]: A sequence of IOIs that loops (e.g., [0.5, 1.0, 0.75]).
        start_time (float, optional): 
            The timestamp of the first onset. Defaults to 0.0.
        num_events (int, optional): 
            The total number of onsets to generate. Defaults to 10.
        tolerance (Optional[float], optional): 
            The maximum random deviation to apply to each IOI for a more "human" feel.
            The deviation is a random value between [-tolerance, +tolerance].
            If None, the timing is perfectly precise. Defaults to None.

    Returns:
        np.ndarray: An array of the generated onset timestamps.
    """
    if num_events <= 0:
        return np.array([])

    onset_times = [start_time]
    current_time = start_time

    # Loop to generate the remaining (num_events - 1) onsets
    for i in range(num_events - 1):
        # Determine the base IOI for this step
        if isinstance(pattern, float):
            # Case 1: The pattern is a single, constant IOI
            ioi = pattern
        else:
            # Case 2: The pattern is a list, so we loop through it
            pattern_index = i % len(pattern)
            ioi = pattern[pattern_index]
        
        # Apply random tolerance if specified
        if tolerance is not None and tolerance > 0:
            deviation = random.uniform(-tolerance, tolerance)
            ioi += deviation

        # Calculate and store the next onset time
        next_time = current_time + ioi
        onset_times.append(next_time)
        current_time = next_time
        
    return np.array(onset_times)