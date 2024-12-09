"""
@author: L.C.M. Cramer
@editor: J.J.A. Vaes

Created on 15-02-2024 12:01:35
Edited on 08-12-2024 16:43:00 
"""

import csv
import re
import os
import numpy as np
from nptdms import TdmsFile
from pandas import DataFrame, to_datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tdms_file import info_from_name, read_lvm, read_tdms,  plot_data
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from matplotlib.backends.backend_pdf import PdfPages

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def plot_load(time, load, load_filtered, idx):
    plt.figure()
    plt.plot(time/60, load)
    plt.plot(time/60, load_filtered)
    plt.xlabel('Time [min]')
    plt.xlim(0,18)
    plt.ylabel('Load [mN]')
    plt.ylim(-1,14)
    plt.title(f"Dataset {idx+1}: Force exerted by Coil")
    plt.grid(True)    
    
def plot_current(time, current, idx):
    plt.figure()
    plt.plot(time/60, current)
    plt.xlabel('Time [min]')
    plt.xlim(0,18)
    plt.ylabel('Current [A]')
    plt.ylim(0,17)
    plt.title(f"Dataset {idx+1}: Current over time")   
    plt.grid(True)

def correct_drift(time, load, drift_duration=20, plot=True):
    """
    Corrects drift in load cell data by fitting a curve through the first and last drift_duration seconds.

    Parameters:
    - time: numpy array of time values.
    - load: numpy array of load values.
    - drift_duration: Duration (in seconds) for which to consider drift at the beginning and end of the data.
    - plot: If True, plots the original and corrected data.

    Returns:
    - corrected_load: Load data after drift compensation.
    - fitted_drift: The fitted drift curve that was subtracted.
    """

    # Find the indices corresponding to the first and last drift_duration seconds
    start_indices = np.where(time <= drift_duration)[0]
    end_indices = np.where(time >= (time[-1] - drift_duration))[0]

    # Get the data for the first and last drift_duration seconds
    time_drift = np.concatenate([time[start_indices], time[end_indices]])
    load_drift = np.concatenate([load[start_indices], load[end_indices]])

    # Define a linear function for fitting the drift
    def linear_func(t, a, b):
        return a * t + b

    # Fit the linear function to the drift data
    popt, _ = curve_fit(linear_func, time_drift, load_drift)

    # Create the drift compensation curve based on the fitted parameters
    fitted_drift = linear_func(time, *popt)

    # Subtract the drift from the original load data to correct it
    corrected_load = load - fitted_drift

    # Optional: Plotting the original and corrected data
    if plot:
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time/60, load, label='Original Load')
        # plt.plot(time/60, load, label='Original Load', color='blue')
        plt.plot(time/60, fitted_drift, label='Fitted Drift', color='red', linestyle='--')
        plt.title('Original Load with Fitted Drift')
        plt.xlabel('Time [min]')
        plt.ylabel('Load [mN]')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time/60, corrected_load, label='Corrected Load', color='green')
        plt.title('Corrected Load (Drift Compensated)')
        plt.xlabel('Time [min]')
        plt.ylabel('Load [mN]')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

    return corrected_load, fitted_drift

def remove_drift(time_array, load_array):
    """
    Align the first 29 seconds of y-values with the horizontal axis (y=0).
    
    Parameters:
    - time_array: Array of time values (in seconds).
    - load_array: Array of corresponding y-values to align.

    Returns:
    - aligned_load_array: The y-values aligned to the horizontal axis for the first 29 seconds.
    """
    
    # Ensure the time_array is a NumPy array for easier filtering
    time_array = np.array(time_array)
    load_array = np.array(load_array)
    
    # Find indices where the time is within the first 29 seconds
    indices_first_29s = time_array <= 29
    
    # Extract y-values corresponding to the first 29 seconds
    y_first_29s = load_array[indices_first_29s]
    
    # Calculate the average of the y-values in the first 29 seconds
    mean_y_first_29s = np.mean(y_first_29s)
    
    # Subtract the mean from all y-values to align the first 29 seconds with y=0
    aligned_load_array = load_array - mean_y_first_29s
    
    return aligned_load_array

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
             
def calculate_mean_via_time(time, load, interval=30):
    # Find the indexes where time exceeds n*interval for some integer n
    timestamps = []
    for n in range(int(time[-1] / interval) + 1):
        index = np.searchsorted(time, n * interval, side='right')
        if index < len(time):
            timestamps.append(index)

    # Calculate mean load between consecutive timestamps
    mean_load_values = []
    for i in range(len(timestamps) - 1):
        start_index = timestamps[i]
        end_index = timestamps[i + 1]
        mean_load = np.mean(load[start_index:end_index])
        mean_load_values.append(mean_load)

    return mean_load_values

def normalize_mean_load_values(mean_load_values, current_setpoints):
    """
    Normalize mean load values at 10 A.
    Parameters:
        mean_load_values (array-like): Array of mean load values.
        current_setpoints (array-like): Array of corresponding current setpoints.
        
    Returns:
        array-like: Normalized mean load values.
    """
    reference_current = 8
    
    # Find the index corresponding to 10 A
    index_10A = np.where(current_setpoints == reference_current)[0][0]
    mean_load_at_10A = mean_load_values[index_10A]
    
    # Calculate normalization factor
    normalization_factor = reference_current / current_setpoints
    
    # Normalize current values
    normalized_current = current_setpoints * normalization_factor
    
    # Normalize load values (assuming load is directly proportional to current)
    normalized_mean_load_values = mean_load_values * normalization_factor 
    
    return normalized_mean_load_values
    
def process_data_0(result):
    # Process data before specific to your requirements
    # Define useful data
    time_array =  result.loc[12, 'x']  # Assuming 'x' column represents time
    load_EM_array = result.loc[12, 'y'] * (1 / 0.02596344) * (25 / 145)   # Assuming 'y' column represents load data
    time_current_array = result.loc[0, 'x']
    current_array = result.loc[0, 'y']

    return time_array, load_EM_array, time_current_array, current_array

def process_data_1(result):
    time_array =  result.loc[12, 'x']  # Assuming 'x' column represents time
    load_EM_array = result.loc[12, 'y'] * (1 / 0.02596344) * (25 / 145)   # Assuming 'y' column represents load data
    time_current_array = result.loc[0, 'x']
    current_array = result.loc[0, 'y']
    
    return time_array, load_EM_array, time_current_array, current_array

def process_data_2(result):
    time_array =  result.loc[12, 'x']  # Assuming 'x' column represents time
    load_EM_array = result.loc[12, 'y'] * (1 / 0.02596344) * (25 / 145)   # Assuming 'y' column represents load data
    time_current_array = result.loc[0, 'x']
    current_array = result.loc[0, 'y']

    return time_array, load_EM_array, time_current_array, current_array

def process_data_3(result):
    time_array =  result.loc[12, 'x']  # Assuming 'x' column represents time
    load_EM_array = result.loc[12, 'y'] * (1 / 0.02596344) * (25 / 145)   # Assuming 'y' column represents load data
    time_current_array = result.loc[0, 'x']
    current_array = result.loc[0, 'y']

    return time_array, load_EM_array, time_current_array, current_array

def compute_combined_performance(combined_normalized_load):
    #Plot to check spread fits result  
    array_filled_with_10 = np.full_like(combined_normalized_load, 8)
    
    plt.figure()
    plt.scatter(array_filled_with_10,combined_normalized_load)
    plt.ylim(0, 10)
    plt.xlabel('Current [A]')
    plt.ylabel('Normalised Load [mN]')
    plt.title(f'{testname}: Spread of normalised load at 8 ampere')
    plt.grid(True)
    # plt.close()
    
    #compute mean value and sigma
    mean_value = np.mean(combined_normalized_load)
    sigma = 3 * np.std(combined_normalized_load)
    
    print(f"Normalized force = {mean_value:.3f} ± {sigma:.3f} ({sigma/mean_value*100:.3f}%)")

    return mean_value, sigma

def process_and_plot(*results, process_func_0=process_data_0, process_func_1=process_data_1, process_func_2=process_data_2, process_func_3=process_data_3):
    # Initialize lists to store mean load values and current setpoints
    all_mean_load_values_raw = []
    all_mean_load_values = []
    all_current_setpoints = []
    all_normalized_mean_load_10A_raw = []
    all_normalized_mean_load_10A = []
    all_load_drift = []
    all_time_values = []
    all_load_raw = []
    all_load_filtered = []

    # Process each dataset
    for idx, result in enumerate(results):
        # Choose processing function based on index, for some reason data flips between tests
        if idx == 0:
            time_array, load_EM_array, time_current_array, current_array = process_func_0(result)
        elif idx == 1:
            time_array, load_EM_array, time_current_array, current_array = process_func_1(result)
        elif idx == 2:
            time_array, load_EM_array, time_current_array, current_array = process_func_2(result)
        else:
            time_array, load_EM_array, time_current_array, current_array = process_func_3(result)

        drift_load = remove_drift(time_array, load_EM_array)
        filtered_load_data_savgol = filter_data_savgol(load_EM_array)
        drift_load_filtered = remove_drift(time_array, filtered_load_data_savgol)
        
        corrected_load, fitted_drift = correct_drift(time_array, filtered_load_data_savgol, drift_duration=20, plot=True)

        plot_load(time_array, drift_load, drift_load_filtered, idx)
        # plot_current(time_current_array, current_array, idx)
        
        current_setpoints = np.arange(0, 17, 0.5)
        # Call the function with the data 
        mean_load_values_raw = calculate_mean_via_time(time_array, load_EM_array)
        mean_load_values     = calculate_mean_via_time(time_array, corrected_load)
        # print(len(mean_load_values))
        
        #Cut data to correct length 
        current_setpoints    = current_setpoints[0:17]
        mean_load_values_raw = mean_load_values_raw[0:17]
        mean_load_values     = mean_load_values[0:17]
        
        corrected_load = corrected_load[0:31760]
        time_array     = time_array[0:31760]
        load_raw       = drift_load[0:31760]
        load_filtered  = drift_load_filtered[0:31760]

        # Subtract the value of the first element from each element such that 0 weight is 0 load
        mean_load_values_raw -= mean_load_values_raw[0] 
        mean_load_values     -= mean_load_values[0] 

        normalized_mean_load_10A_raw = normalize_mean_load_values(mean_load_values_raw[1:], current_setpoints[1:])
        normalized_mean_load_10A     = normalize_mean_load_values(mean_load_values[1:], current_setpoints[1:])
        
        # Store mean load values and current setpoints
        all_mean_load_values_raw.append(mean_load_values_raw)
        all_mean_load_values.append(mean_load_values)
        all_current_setpoints.append(current_setpoints)
        all_normalized_mean_load_10A_raw.append(normalized_mean_load_10A_raw)
        all_normalized_mean_load_10A.append(normalized_mean_load_10A)

        all_load_drift.append(corrected_load)
        all_time_values.append(time_array)
        all_load_raw.append(load_raw)
        all_load_filtered.append(load_filtered)

        # Fit a linear regression line to raw data
        slope_raw, intercept_raw, r_value_raw, _, _ = linregress(current_setpoints, mean_load_values_raw)

        with open(f'{testname}.txt', "a") as text_file:
            # Print slope and intercept for each dataset
            if idx == 0:
                test = '0'
            elif idx == 1:
                test = '1'
            elif idx == 2:
                test = '2'
            else:
                test = '3'       
            print(f"Dataset {test} (Uncompensated):")
            print(f"Slope [mN/A]: {round(slope_raw, 4)}, Intercept [mN]: {round(intercept_raw, 4)}")
            print(f"R^2 = {r_value_raw**2:.4f}")

            # Calculate standard deviation
            sigma_raw = np.std(normalized_mean_load_10A_raw)
            uncertainty_raw = 3 * sigma_raw / np.sqrt(len(mean_load_values_raw[1:]))
            
            # Print 3 sigma standard deviation
            print(f"Standard Deviation [mN]: {round(sigma_raw, 4)}")
            print(f"Uncertainty per value is ± {round(uncertainty_raw, 4)} [mN]")

            # Fit a linear regression line
            slope, intercept, r_value, _, _ = linregress(current_setpoints, mean_load_values)

            # Print slope and intercept for each dataset
            if idx == 0:
                test = '0'
            elif idx == 1:
                test = '1'
            elif idx == 2:
                test = '2'
            else:
                test = '3'       
            print(f"Dataset {test} (Compensated):")
            print(f"Slope [mN/A]: {round(slope, 4)}, Intercept [mN]: {round(intercept, 4)}")
            print(f"R^2 = {r_value**2:.4f}")
            
            print(f"Dataset {test} (Uncompensated):", file=text_file)
            print(f"Slope [mN/A]: {round(slope_raw, 4)}, Intercept [mN]: {round(intercept_raw, 4)}", file=text_file)
            print(f"R^2 = {r_value_raw**2:.4f}", file=text_file)

            print(f"Standard Deviation [mN]: {round(sigma_raw, 4)}", file=text_file)
            print(f"Uncertainty per value is ± {round(uncertainty_raw, 4)} [mN]", file=text_file)
            print("\n", file=text_file)  # Add a newline for readability between datasets


            print(f"Dataset {test} (Compensated):", file=text_file)
            print(f"Slope [mN/A]: {round(slope, 4)}, Intercept [mN]: {round(intercept, 4)}", file=text_file)
            print(f"R^2 = {r_value**2:.4f}", file=text_file)

            # Calculate standard deviation
            sigma = np.std(normalized_mean_load_10A)
            uncertainty = 3 * sigma / np.sqrt(len(mean_load_values[1:]))
            
            # Print 3 sigma standard deviation
            print(f"Standard Deviation [mN]: {round(sigma, 4)}")
            print(f"Uncertainty per value is ± {round(uncertainty, 4)} [mN]")

            print(f"Standard Deviation [mN]: {round(sigma, 4)}", file=text_file)
            print(f"Uncertainty per value is ± {round(uncertainty, 4)} [mN]", file=text_file)
            print("\n", file=text_file)  # Add a newline for readability between datasets
            print("\n", file=text_file)  # Add a newline for readability between datasets
        
    #Convert lists to numpy arrays for easier manipulation
    all_mean_load_values  = np.array(all_mean_load_values)
    all_current_setpoints = np.array(all_current_setpoints)
    all_load_drift        = np.array(all_load_drift)
    all_time_values       = np.array(all_time_values)
    all_load_raw          = np.array(all_load_raw)
    all_load_filtered     = np.array(all_load_filtered)

    combined_normalized_load_raw = np.concatenate(all_normalized_mean_load_10A_raw)
    mean_value_raw, Csigma_raw = compute_combined_performance(combined_normalized_load_raw)
    combined_normalized_load = np.concatenate(all_normalized_mean_load_10A)
    mean_value, Csigma = compute_combined_performance(combined_normalized_load)

    with open(f'{testname}.txt', "a") as text_file:
        print(f"Normalized force (Uncompensated) = {mean_value_raw:.3f} ± {Csigma_raw:.3f} ({Csigma_raw/mean_value_raw*100:.3f}%)", file=text_file)
        print(f"Normalized force (Compensated)   = {mean_value:.3f} ± {Csigma:.3f} ({Csigma/mean_value*100:.3f}%)", file=text_file)

    # Define markers for each dataset
    markers = ['o', 's', '^', '*']
    trendline_colors = ['red', 'purple', 'brown', 'yellow']

    # Compute linear regression for each dataset and plot
    plt.figure()
    for i in range(len(results)):
        if i == 0:
            test = '0'
            marker='o'
        elif i == 1:
            test = '1'
            marker='x'
        elif i == 2:
            test = '2'
            marker='s'
        else:
            test = '3'
            marker='*'
        slope, intercept, r_value, _, _ = linregress(all_current_setpoints[i], all_mean_load_values_raw[i])
        # marker = markers[i % len(markers)]  # Cycling through markers if more than 4 datasets
        plt.plot(all_current_setpoints[i], all_mean_load_values_raw[i], marker=markers[i], linestyle='-', label=f'Dataset {test}')        
        plt.plot(all_current_setpoints[i], intercept + slope * np.array(all_current_setpoints[i]), linestyle='--', color=trendline_colors[i], label=f'Linear Fit: y = {slope:.3f}x + {intercept:.3f}\nR$^2$ = {r_value**2:.3f}')

    # Customize plot
    plt.legend()
    plt.xlabel('Current Setpoint [A]')
    plt.ylabel('Load [mN]')
    plt.title(f'{testname}: {calname}')
    plt.grid(True)

    # Compute linear regression for each dataset and plot
    plt.figure()
    for i in range(len(results)):
        if i == 0:
            test = '0'
            marker='o'
        elif i == 1:
            test = '1'
            marker='x'
        elif i == 2:
            test = '2'
            marker='s'
        else:
            test = '3'
            marker='*'
        slope, intercept, r_value, _, _ = linregress(all_current_setpoints[i], all_mean_load_values[i])
        # marker = markers[i % len(markers)]  # Cycling through markers if more than 4 datasets
        plt.plot(all_current_setpoints[i], all_mean_load_values[i], marker=markers[i], linestyle='-', label=f'Dataset {test}')        
        plt.plot(all_current_setpoints[i], intercept + slope * np.array(all_current_setpoints[i]), linestyle='--', color=trendline_colors[i], label=f'Linear Fit: y = {slope:.3f}x + {intercept:.3f}\nR$^2$ = {r_value**2:.3f}')

    # Customize plot
    plt.legend()
    plt.xlabel('Current Setpoint [A]')
    plt.ylabel('Load [mN]')
    plt.title(f'{testname}: {calname}')
    plt.grid(True)

    plt.figure()
    for i in range(len(results)):
        if i == 0:
            test = '0'
            marker='o'
        elif i == 1:
            test = '1'
            marker='x'
        elif i == 2:
            test = '2'
            marker='s'
        else:
            test = '3'
            marker='*'
        plt.plot(all_time_values[i]/60, all_load_filtered[i], linestyle='-', label=f'Dataset {test}')            
    plt.legend()
    plt.xlabel('Time [min]')
    plt.ylabel('Load [mN]')
    plt.title(f'{testname}: {calname}')
    plt.grid(True)

    plt.figure()
    for i in range(len(results)):
        if i == 0:
            test = '0'
            marker='o'
        elif i == 1:
            test = '1'
            marker='x'
        elif i == 2:
            test = '2'
            marker='s'
        else:
            test = '3'
            marker='*'
        plt.plot(all_time_values[i]/60, all_load_drift[i], linestyle='-', label=f'Dataset {test}')        
    plt.legend()
    plt.xlabel('Time [min]')
    plt.ylabel('Load [mN]')
    plt.title(f'{testname}: {calname}')
    plt.grid(True)

    plt.figure()
    for i in range(len(results)):
        if i == 0:
            test = '0'
            marker='o'
        elif i == 1:
            test = '1'
            marker='x'
        elif i == 2:
            test = '2'
            marker='s'
        else:
            test = '3'
            marker='*'
        plt.plot(all_time_values[i]/60, all_load_raw[i], linestyle='-', label=f'Dataset {test} (Unfiltered)')            
        plt.plot(all_time_values[i]/60, all_load_filtered[i], linestyle='-', label=f'Dataset {test} (Filtered)')            
    plt.legend()
    plt.xlabel('Time [min]')
    plt.ylabel('Load [mN]')
    plt.title(f'{testname}: {calname}')
    plt.grid(True)

    multipage(f'{testname}.pdf')
    plt.show()

    return all_current_setpoints, all_mean_load_values 


# Example datasets
results0 = read_tdms(r"Calibrations\20240924 Cal_Full_Vac_200\104222.TDMS")
results1 = read_tdms(r"Calibrations\20240924 Cal_Full_Vac_200\114130.TDMS")
results2 = read_tdms(r"Calibrations\20240924 Cal_Full_Vac_200\115937.TDMS")

testname = "CAL-111a"
calname  = "Full test bench in vacuum (200 \u00B0C)"
print(testname)
print(results0)

# Check if the file exists, and delete it if it does
if os.path.exists(f'{testname}.txt'):
    os.remove(f'{testname}.txt')  # This deletes the file

# Process and plot both datasets together
current_setpoint, mean_load_values = process_and_plot(results0, results1, results2)