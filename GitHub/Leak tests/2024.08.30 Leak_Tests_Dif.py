"""
@author: J.J.A. Vaes

Created on 08-12-2024 16:43:00
"""

import csv
import re
import pandas as pd
import numpy as np
from nptdms import TdmsFile
from pandas import DataFrame, to_datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tdms_file import info_from_name, read_lvm, read_tdms,  plot_data
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.optimize import curve_fit
from statistics import mean

import CoolProp
from CoolProp.State import AbstractState

# Define the linear function to fit
def linear_func(x, a, b):
    return a * x + b

def plot_pressure(time,pressure, idx):
    plt.figure()
    plt.plot(time, pressure)
    plt.xlabel('time [s]')
    plt.ylabel('Pressure [mbar]')
    plt.grid()
    # plt.ylim(-5, 16)
    # plt.xlim(-20,2300)
    plt.title(f"Dataset {idx+1}: Pressure drop over time")
    # plt.close()   

def filter_data_butterworth(data, fs=30.0, cutoff_freq=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def filter_data_savgol(data):
    """
    Apply Savitzky-Golay filter with a polynomial order of 1 and a window size of 1 second.

    Parameters:
        data (array_like): The input data array.
        sampling_rate (float): The sampling rate of the data in Hz.

    Returns:
        array_like: The filtered data array.
    """
    # Calculate the window size in number of samples
    sampling_rate = 30 #Hz
    window_size_samples = int(sampling_rate)
    
    # Ensure the window size is odd
    if window_size_samples % 2 == 0:
        window_size_samples += 1
    
    # Apply Savitzky-Golay filter with specified parameters
    filtered_data = savgol_filter(data, window_size_samples, polyorder=1)
    
    return filtered_data

# Define a function to process and plot data
def process_and_plot(*results):
    OG_pressure_values_S = []
    OG_time_values_S     = []
    all_pressure_values = []
    all_time_values     = []
    OG_pressure_values  = []
    OG_time_values      = []
    all_pressure_ambient= []

    # Process each dataset
    for idx, result in enumerate(results):
        # Define useful data    
        time_array_pressure = result.loc[2, 'x']
        pressure_array = result.loc[2, 'y']
        plot_pressure(time_array_pressure, pressure_array, idx)
        pressure_ambient = mean(pressure_array[0:50])
        print(f"Ambient pressure is: {pressure_ambient:.3f} [mbar]")

        # # Make sure to have equal lenghts.
        # if len(pressure_array) <= 700:
        #     OGpressure_S = pressure_array[0:600]
        #     OGtime_S     = time_array_pressure[0:600]
        #     # print('tttt', OGpressure_S)
        #     OG_pressure_values_S.append(OGpressure_S)
        #     OG_time_values_S.append(OGtime_S)
        # else:
        #     pressure_array = pressure_array[100:6000]
        #     time_array_pressure = time_array_pressure[100:6000]
        #     print('ssss', pressure_array)
        #     all_pressure_values.append(pressure_array)
        #     all_time_values.append(time_array_pressure)
        # return OG_pressure_values_S, OG_time_values_S, all_pressure_values, all_time_values

        # Save original data.
        # OGpressure_S = pressure_array[0:600]
        # OGtime_S     = time_array_pressure[0:600]
        OGpressure = pressure_array[0:6000]
        OGtime     = time_array_pressure[0:6000]

        # Make sure to have equal lenghts.
        pressure_array = pressure_array[100:6000]
        time_array_pressure = time_array_pressure[100:6000]

        all_pressure_values.append(pressure_array)
        all_time_values.append(time_array_pressure)
        # OG_pressure_values_S.append(OGpressure_S)
        # OG_time_values_S.append(OGtime_S)
        OG_pressure_values.append(OGpressure)
        OG_time_values.append(OGtime)

        all_pressure_ambient.append(pressure_ambient)

        # Plot load data with Butterworth filter 
        filtered_pressure_data_butterworth = filter_data_butterworth(pressure_array)
    
        # Plot load data with Savitzky-Golay filter 
        filtered_pressure_data_savgol = filter_data_savgol(pressure_array)

    # all_pressure_values_S = np.array(all_pressure_values_S)
    # all_time_values_S = np.array(all_time_values_S)
    all_pressure_values = np.array(all_pressure_values)
    all_time_values = np.array(all_time_values)
    # OG_Pressure_S = np.array(OG_pressure_values_S)
    # OG_Time_S     = np.array(OG_time_values_S)
    OG_Pressure = np.array(OG_pressure_values)
    OG_Time     = np.array(OG_time_values)

    all_pressure_ambient = np.array(all_pressure_ambient)

    # Define the pressure difference to find the local slope
    target_pressure_differences = [1500, 1250, 1000, 750, 500]
    tolerance = 20  # Define a tolerance range to consider

    # List to store slope values
    slope_values = {target: [] for target in target_pressure_differences}

    # Identify intervals where pressure difference is within the tolerance range
    interval_length = 0.201  # 0.225-second interval
    for target_pressure_difference in target_pressure_differences:
        for start_time in np.arange(all_time_values.min(), all_time_values.max() - interval_length, 0.1):  # Moving window of 0.225 seconds
            end_time = start_time + interval_length
            mask = (all_time_values >= start_time) & (all_time_values < end_time)
            time_filtered = all_time_values[mask]
            pressure_filtered = all_pressure_values[mask]

            # Check if the pressure difference within the interval is close to the target
            if len(time_filtered) > 1:
                # pressure_diff = np.max(pressure_filtered) - np.min(pressure_filtered)
                pressure_diff = np.max(pressure_filtered) - pressure_ambient
                if abs(pressure_diff - target_pressure_difference) <= tolerance:
                    # Perform the linear fit
                    popt, _ = curve_fit(linear_func, time_filtered, pressure_filtered)
                    a, _ = popt
                    slope_values[target_pressure_difference].append(a)

                    # Plot the results for each interval
                    plt.figure(figsize=(10, 6))
                    plt.scatter(time_filtered, pressure_filtered, label='Data')
                    plt.plot(time_filtered, linear_func(time_filtered, *popt), color='red', label=f'Fitted line: {a:.3f} [mbar/s]')
                    plt.xlabel('Time [s]')
                    plt.ylabel('Pressure [mbar]')
                    plt.title(f'Pressure Drop with Linear Fit ({start_time:.1f}s to {end_time:.1f}s)')
                    plt.legend()
                    # plt.show()
                    plt.close()

    # Calculate the average slope value for each target pressure difference
    average_slopes = {target: np.mean(slopes) if slopes else None for target, slopes in slope_values.items()}
    massflows = []

    for target, avg_slope in average_slopes.items():
        if avg_slope is not None:
            print(f"Average slope value at {target} mbar pressure difference: {avg_slope:.3f} [mbar/s]")

            fit_massflow = avg_slope * 2.430 * 28.014 / (8.314 * (21.47+273.15) * 10)
            print(f"Average leak rate at {target} mbar pressure difference: {fit_massflow:.3f} [mg/s]")
            massflows.append(fit_massflow)
        else:
            print(f"No intervals found with a pressure difference close to {target} mbar")
    
    avg_massflow = (sum(massflows) / len(massflows))
    print(f"Average massflow is: {avg_massflow:.3f} [mg/s]")

    # Optionally, convert results to a DataFrame for easier viewing
    fit_results_df = pd.DataFrame({
        'Target Pressure Difference': target_pressure_differences,
        'Average Slope': [average_slopes[target] for target in target_pressure_differences]
    })
    
    pressure_nitrogen = 1e5
    temperature_nitrogen = 300 + 273.15
    nitrogen = AbstractState("HEOS", "nitrogen")
    nitrogen.update(CoolProp.PT_INPUTS, pressure_nitrogen, temperature_nitrogen)
    nitrogen_viscosity = nitrogen.viscosity()
    nitrogen_molarmass = nitrogen.molar_mass()
    print('nitrogen viscosity',nitrogen_viscosity)
    print('nitrogen molas mass',nitrogen_molarmass)   

    pressure_water = 1e5
    temperature_water = 300 + 273.15
    water = AbstractState("HEOS", "water")
    water.update(CoolProp.PT_INPUTS, pressure_water, temperature_water)
    water_viscosity = water.viscosity()
    water_molarmass = water.molar_mass()
    print('water viscosity',water_viscosity)
    print('water molas mass',water_molarmass)

    mlwater = (0.0357939 * nitrogen_viscosity * water_molarmass) / (water_viscosity * 8.314 * temperature_water)    
    print('leak rate water [mg/s]: ', mlwater*100000)
    
    # Define markers for each dataset
    markers = ['o', 's', '^', '*']
    trendline_colors = ['red', 'purple', 'brown', 'yellow']
    linestyles = ['-', '--', '-.']

    # Compute linear regression for each dataset and plot
    plt.figure()
    for i in range(len(results)):
        marker = markers[i % len(markers)]  # Cycling through markers if more than 4 datasets
        # plt.plot(all_time_values[i], all_pressure_values[i], linestyle='-', label=f'Dataset {i+1}')
        plt.plot(OG_Time[i]/60, OG_Pressure[i], linestyle=linestyles[i], label=f'Dataset {i+1}')

    # Customize plot
    plt.legend(fontsize='small', loc='upper right')
    plt.xlabel('Time [min]')
    plt.ylabel('Pressure [mbar]')
    plt.title('VLM-JV1: Pressure drop over time')
    plt.xlim([0, 10]) 
    plt.ylim([1000, 2425])
    plt.grid(True)
    plt.legend()

    plt.figure()
    for i in range(len(results)):
        marker = markers[i % len(markers)]  # Cycling through markers if more than 4 datasets
        # Compute the derivative of pressure with respect to time
        dP_dt  = np.gradient(-all_pressure_values[i], all_time_values[i])  # this gives dP/dt
        dmg_dt = np.gradient(-all_pressure_values[i]* 2.430 * 28.014 / (8.314 * (24.38+273.15) * 10), all_time_values[i])
        dmg_dt_filtered = filter_data_savgol(dmg_dt)
        # Plot the derivative
        # plt.plot(all_time_values[i], dP_dt, linestyle=linestyles[i], label=f'Dataset {i+1}')
        plt.plot(all_pressure_values[i]-all_pressure_ambient[i], dmg_dt_filtered,  linestyle=linestyles[i],  label=f'Dataset {i+1}')

    # Customize plot
    plt.legend(fontsize='small', loc='upper right')
    plt.xlabel('Pressure difference [mbar]')
    plt.ylabel('Leak rate [mg/s]')
    plt.title('VLM-JV1: Leak rate over pressure difference')
    plt.xlim([0, 1200]) 
    plt.ylim([0, 0.35])
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

    return all_time_values, all_pressure_values 

# Example datasets
results0 = read_tdms(r"Leak tests\2024.08.30 test01\135541.TDMS")
results1 = read_tdms(r"Leak tests\2024.08.30 test01\141925.TDMS")

# Process and plot both datasets together
process_and_plot(results0, results1)