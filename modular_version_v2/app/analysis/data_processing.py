# File: app/analysis/data_processing.py
import pandas as pd
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt


def apply_data_section(df: pd.DataFrame, t_min_str: str, t_max_str: str) -> pd.DataFrame:
    """Slices the DataFrame to a specified time interval."""
    try:
        t_min = float(t_min_str)
        t_max = float(t_max_str)
        if t_min < t_max:
            return df[(df['TIME'] >= t_min) & (df['TIME'] <= t_max)].copy()
    except (ValueError, KeyError):
        # If input is invalid or 'TIME' column is missing, return original df
        return df
    return df


def apply_tukey_window(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Applies a Tukey window to all load columns in the DataFrame."""
    if 'TIME' not in df.columns or len(df) <= 1:
        return df

    df_windowed = df.copy()
    window = tukey(len(df_windowed), alpha)
    load_cols = [c for c in df_windowed.columns if c not in ['TIME', 'FREQ', 'NO', 'DataFolder']]
    df_windowed.loc[:, load_cols] = df_windowed.loc[:, load_cols].multiply(window, axis=0)
    return df_windowed


def apply_low_pass_filter(df: pd.DataFrame, column: str, cutoff: float, order: int) -> pd.DataFrame:
    """Applies a low-pass Butterworth filter to a specific column in the DataFrame."""
    df_filtered = df.copy()
    try:
        # Calculate sampling frequency
        fs = 1 / df_filtered.index.to_series().diff().mean()

        # Perform filtering
        b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
        df_filtered[column] = filtfilt(b, a, df_filtered[column])

    except (ValueError, ZeroDivisionError) as e:
        print(f"Could not apply filter: {e}")
        return df  # Return original DataFrame on error

    return df_filtered