import os
import numpy as np

class Config:
    ENVELOPE_MODE: str = 'HILBERT'
    MIN_POINTS: int = 3
    THRESHOLD_MODE: str = 'MANUAL'
    THRESHOLD_VALUE: float = 1.4
    THRESHOLD_K: float = 1.0
    
    START_IDX: int = 1
    END_IDX: int = 21
    USE_PARALLEL: bool = True
    N_JOBS: int = max(1, (os.cpu_count() or 4) - 1)
    
    PLOT_CONFIDENCE_INTERVAL: bool = True

def calculate_iddr_feature(signal: np.ndarray, 
                           fs: float, 
                           min_points: int, 
                           envelope_mode: str,
                           forced_threshold: float) -> float:
    if signal.size < min_points * 2:
        return 0.0

    sig = signal - np.mean(signal)

    if envelope_mode.upper() == 'HILBERT':
        try:
            from scipy.signal import hilbert
            analytic_signal = hilbert(sig)
            envelope = np.abs(analytic_signal)
        except Exception:
            envelope = np.abs(sig)
    else:
        envelope = np.abs(sig)

    threshold = forced_threshold
    
    max_val = np.max(envelope)
    if max_val <= threshold:
        return 0.0

    is_burst = envelope > threshold
    diff = np.diff(np.concatenate(([0], is_burst.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    iddr_accum = 0.0
    valid_events = 0

    for s, e in zip(starts, ends):
        duration_points = e - s
        
        if duration_points < min_points:
            continue

        burst_env = envelope[s:e]
        peak_amp = np.max(burst_env)
        peak_idx_local = np.argmax(burst_env) 

        rise_time = (peak_idx_local + 1) / fs
        duration = duration_points / fs

        term = rise_time * duration
        if term > 1e-20: 
            event_iddr = peak_amp / term
            iddr_accum += event_iddr 
            valid_events += 1

    if valid_events == 0:
        return 0.0
    
    return float(iddr_accum)