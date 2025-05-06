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

# Define the folder path and date range
folder_path = 'data_phase'
start_date = '24-04-2025'
end_date = '06-05-2025'
channels = [1, 2, 3]  # Specify the channels you want to load

# Call the function to load phase data
df_phase, df_freq = aux.load_pc_data(folder_path, start_date, end_date, channels)

# Print the DataFrame
print(df_phase)

aux.plot_phase_data(df_phase, channels, '10/100 MHz link')

#%%
# Target datetime value
t_start = datetime(2025, 4, 24, 15, 0)

# 1. Filter phase and frequency data
df_phase_filtered = aux.data_start_time(df_phase, t_start)
df_freq_filtered = aux.data_start_time(df_freq, t_start)

# 2. Run plots + Allan analysis
phase_allan_results = aux.plot_phase_and_allan(df_phase_filtered, channels, sample_rate=1.0)

offsets = {1: 1e5, 2: 2e5, 3: -1e5}
freq_allan_results = aux.plot_frequency_and_allan(df_freq_filtered, channels, sample_rate=1.0)
