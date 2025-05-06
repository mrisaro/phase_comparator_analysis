# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:58:47 2025
main data analysis phase comparator
@author: mar
"""
import numpy as np
import matplotlib.pyplot as plt
import auxiliary as aux
import allantools
from datetime import datetime
from prettytable import PrettyTable

# Define the folder path and date range
folder_path = 'data_phase'
start_date = '24-04-2025'
end_date = '06-05-2025'

f0 = 100e6
channels = [1, 2, 3]  # Specify the channels you want to load

# Call the function to load phase data
df_phase, df_freq = aux.load_pc_data(folder_path, start_date, end_date, channels)

# Print the DataFrame
print(df_phase)

#%%
aux.plot_raw_phase(df_phase, channels, '10/100 MHz link')

# Target datetime value
t_start = datetime(2025, 4, 24, 15, 0)

# 1. Filter phase and frequency data
df_phase = aux.data_start_time(df_phase, t_start)
df_freq = aux.data_start_time(df_freq, t_start)

# 2. Run plots + Allan analysis
phase_allan_results = aux.plot_phase_and_allan(df_phase, channels, sample_rate=1.0)

offsets = {1: 0, 2: 1e-3, 3: -1e-3}
fr_allan = aux.plot_frequency_and_allan(df_freq, channels, sample_rate=1.0, offsets=offsets)

#%% Testing glitche removal

# - df_freq → frequency DataFrame
# - f0 → nominal frequency
# - freq_allan_results → result dict from plot_frequency_and_allan()
# - channels → list of channels

# Create PrettyTable object
table = PrettyTable()
table.field_names = ["Channel", "Rel. Mean Δf/f₀", "Last Allan Dev"]

for channel in channels:
    # Compute relative mean frequency offset
    freq_values = df_freq[f'Freq_{channel}'].values
    delta_f = np.mean(freq_values - f0) / f0

    # Get last Allan deviation point
    taus, adevs = fr_allan[channel]
    last_adev = adevs[-1]/f0

    # Add row to table
    table.add_row([f'Ch {channel}', f'{delta_f:.2e}', f'{last_adev:.2e}'])

# Print to terminal
print("\nSummary of Frequency Analysis:")
print(table)
