# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:45:10 2025
auxiliary functions for phase comparator
@author: mar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import allantools
from datetime import datetime, timedelta

# ------------------- Load files and data ------------------- #

def load_pc_data(folder_path, start_date_str, end_date_str, channels, nominal_freq=100e6):
    """
    Load files from the TimeTech phase comparator.

    Parameters:
        folder_path (str): where files are located.
        start_date_str (str): starting date of the data (dd-mm-yyyy).
        stop_date_str (str): stoping date of the data (dd-mm-yyyy).
        channels (list): channels that have available data, [1,2,3] or [4,5]
        
    Returns:
        df_phase (pd.df): a dataframe with columns with time and phase channels
        df_freq (pd.df): a dataframe with columns with time and frequency channels
    """
    start_date = datetime.strptime(start_date_str, '%d-%m-%Y')
    end_date = datetime.strptime(end_date_str, '%d-%m-%Y')

    phase_data = {channel: [] for channel in channels}
    time_data = []

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%d-%m-%Y')
        filename = f'phase_pco_{date_str}.dat'
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                data = np.loadtxt(file_path, dtype=str)
                time_data_str = data[:, 0]
                time_data.extend([datetime.strptime(t, '%d/%m/%Y/%H:%M:%S.%f') for t in time_data_str])

                for channel in channels:
                    phase_data[channel].extend(data[:, channel].astype(float))
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

        current_date += timedelta(days=1)

    for channel in channels:
        phase_data[channel] = np.array(phase_data[channel])

    time_data = np.array(time_data)
    time_seconds = np.array([(t - time_data[0]).total_seconds() for t in time_data])

    # PHASE DATAFRAME
    df_phase = pd.DataFrame({'Time': time_data})
    for channel in channels:
        df_phase[f'Ch_{channel}'] = phase_data[channel]

    # FREQUENCY DATAFRAME
    df_freq = pd.DataFrame({'Time': time_data[1:]})  # time shortened by 1 due to diff
    dt = np.diff(time_seconds)
    for channel in channels:
        dphase = np.diff(phase_data[channel])
        freq = nominal_freq + (dphase / dt) * nominal_freq
        df_freq[f'Freq_{channel}'] = freq

    return df_phase, df_freq

# ------------------- Plot raw data in phase ------------------- #

def plot_raw_phase(df_phase, channels, name):
    """
    Plot raw phase data of all channels before filtering.

    Parameters:
    - df_phase: DataFrame with phase data
    - channels: list of channel numbers
    """
    plt.figure(figsize=(10,6), num=name)
    for channel in channels:
        phase_ps = df_phase[f'Ch_{channel}'].to_numpy() * 1e12  # convert to picoseconds
        plt.plot(df_phase['Time'], phase_ps, label=f'Ch {channel}')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Phase (ps)', fontsize=12)
    plt.title('Raw Phase Data (Before Filtering)', fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------- Plot phase data and stability ------------------- #

def plot_phase_and_allan(df_phase, channels, sample_rate=1.0):
    fig, axs = plt.subplots(2,1,figsize=(10,8), num='phase + allan')

    allan_results = {}

    # Upper plot: phase
    for channel in channels:
        phase_ps = df_phase[f'Ch_{channel}'].values * 1e12  # seconds → ps
        axs[0].plot(df_phase['Time'], phase_ps, label=f'Ch {channel}')
        
        # Calculate Allan deviation
        taus, adevs, _, _ = allantools.oadev(df_phase[f'Ch_{channel}'].values, 
                                             rate=sample_rate, data_type='phase')
        allan_results[channel] = (taus, adevs)
        
        # Lower plot: Allan deviation
        axs[1].loglog(taus, adevs, '-o', label=f'Allan Dev Ch {channel}')

    axs[0].set_xlabel(r'Time', fontsize=12)
    axs[0].set_ylabel(r'$\varphi$ (ps)', fontsize=12)
    axs[0].grid(which='both', linestyle='--')
    axs[0].legend()

    axs[1].set_xlabel(r'$\tau$ (s)', fontsize=12)
    axs[1].set_ylabel(r'OADEV', fontsize=12)
    axs[1].grid(which='both', linestyle='--')
    axs[1].legend()

    fig.tight_layout()
    plt.show()

    return allan_results

# ------------------- Plot freq data and stability ------------------- #

def plot_frequency_and_allan(df_freq, channels, sample_rate=1.0, offsets=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), num='frequency + allan')

    allan_results = {}

    # Handle optional offsets
    if offsets is None:
        offsets = {channel: 0 for channel in channels}
    else:
        # Make sure all channels have an offset, defaulting to 0 if missing
        offsets = {channel: offsets.get(channel, 0) for channel in channels}

    # Upper plot: frequency deviation (Hz)
    for channel in channels:
        freq_dev = df_freq[f'Freq_{channel}'].to_numpy() - 100e6  # deviation from nominal
        freq_dev += offsets[channel]  # apply offset
        axs[0].plot(df_freq['Time'], freq_dev, label=f'Ch {channel}')
        
        # Calculate Allan deviation
        taus, adevs, _, _ = allantools.oadev(freq_dev, rate=sample_rate, data_type='freq')
        allan_results[channel] = (taus, adevs)
        
        # Lower plot: Allan deviation
        axs[1].loglog(taus, adevs, '-o', label=f'Allan Dev Ch {channel}')

    axs[0].set_xlabel(r'Time', fontsize=12)
    axs[0].set_ylabel(r'Frequency Deviation (Hz)', fontsize=12)
    axs[0].grid(which='both', linestyle='--')
    axs[0].legend()

    axs[1].set_xlabel(r'$\tau$ (s)', fontsize=12)
    axs[1].set_ylabel(r'OADEV', fontsize=12)
    axs[1].grid(which='both', linestyle='--')
    axs[1].legend()

    fig.tight_layout()
    plt.show()

    return allan_results

def data_start_time(df, t_start):
    """
    Filters the DataFrame to include only rows after t_start.

    Parameters:
    - df: DataFrame with 'Time' column
    - t_start: datetime object

    Returns:
    - Filtered DataFrame
    """
    return df[df['Time'] >= t_start].reset_index(drop=True)

# ------------------- Glitches detection ------------------- #

def remove_glitches(df, column, window_size=50, threshold=5):
    """
    Remove glitches and return cleaned data + mask.

    Parameters:
    - df: DataFrame
    - column: column name
    - window_size: rolling window size
    - threshold: std multiplier

    Returns:
    - cleaned_series: numpy array (glitches interpolated)
    - mask: boolean numpy array (True = valid, False = glitch)
    """
    series = df[column]
    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_std = series.rolling(window=window_size, center=True).std()

    mask = np.abs(series - rolling_median) < threshold * rolling_std

    cleaned_series = series.copy()
    cleaned_series[~mask] = np.nan
    cleaned_series = cleaned_series.interpolate()

    return cleaned_series.values, mask.values