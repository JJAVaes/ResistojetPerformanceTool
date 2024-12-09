"""
@author: L.C.M. Cramer
@editor: J.J.A. Vaes

Created on 15-02-2024 12:01:35
Edited on 08-12-2024 16:43:00 
"""

import csv
import re
import pandas as pd
import numpy as np
from nptdms import TdmsFile
from pandas import DataFrame, to_datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import butter, filtfilt, savgol_filter
from Plotting_function import *
from Correctionfactors import correction_factor
from tdms_file import info_from_name, read_lvm, read_tdms,  plot_data
from itertools import chain
from scipy.optimize import curve_fit

''' Quadratic function definition '''
# Define the function to fit (quadratic function: ax^2 + bx + c)
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

# Define function to fit a quadratic curve
def fit_quadratic(time, loads):
    popt, _ = curve_fit(quadratic_function, time, loads)
    return popt

# Define a function to fit and plot the data
def fit_and_plot(time_data, load_data, label):
    # Fit the curve to the data
    popt, pcov = curve_fit(quadratic_function, time_data, load_data)

    # Get the optimized parameters
    a_opt, b_opt, c_opt = popt

    # Plot the original data
    plt.figure()
    plt.scatter(time_data, load_data, label='Original data')

    # Plot the fitted curve
    x_fit = np.linspace(min(time_data), max(time_data), 100)
    y_fit = quadratic_function(x_fit, a_opt, b_opt, c_opt)
    plt.plot(x_fit/60, y_fit, color='red', label='Fitted curve')
    plt.xlabel('Time [min]')
    plt.ylabel('Load [mN]')
    plt.title(f'Curve Fitting - {label}')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time_data)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.show()

    # Print the optimized parameters
    #print(f'Optimized parameters for {label}: a = {a_opt}, b = {b_opt}, c = {c_opt}')

# Subtract the "Thruster Off" function from the "Cut Thruster On" function
def subtract_functions(x, a_on, b_on, c_on, a_off, b_off, c_off):
    return quadratic_function(x, a_on, b_on, c_on) - quadratic_function(x, a_off, b_off, c_off)

def Remove_drift(load_thruster_array, loads_data_closed, time_load_array, time_loads_closed):                #(load_thruster_array, loads_data_open, loads_data_closed, cut_loads_data_open, time_load_array, cut_time_loads_open, time_loads_open, time_loads_closed):
    # testname = 'H2O Test 400°C 1.0 Bar:'
    # Define a list of pastel colors
    pastel_colors = ['#FF9999', '#66CCCC', '#99CC99', '#FFCC99', '#CCCCFF', '#FFCCFF', '#FFFF99', '#CCFF99']
    
    # Fit quadratic functions to "Thruster Off" and "Cut Thruster On" data
    popt_closed = fit_quadratic(time_loads_closed, loads_data_closed)
    
    # Get the optimized parameters for both datasets
    a_closed, b_closed, c_closed = popt_closed
    
    # Plot the fitted curve
    x_fit = np.linspace(min(time_load_array), max(time_load_array), 100)
    y_fit = quadratic_function(x_fit, a_closed, b_closed, c_closed)
    
    # Apply filter 
    filtered_loads_data_savgol = filter_data_savgol(load_thruster_array)
    
    # Plot filtered and unfiltered load data for the current test
    plt.figure()
    plt.plot(time_load_array/60, load_thruster_array, label='Thrust (Unfiltered)')
    plt.plot(time_load_array/60, filtered_loads_data_savgol, label='Thrust (Filtered)')
    plt.plot(x_fit/60, y_fit, color='red', linestyle = '--', label='Fitted curve')
    plt.xlabel('Time [min]')
    plt.ylabel('Thrust [mN]')
    plt.title(f'{testname}: Thrust with fitted curve')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time_load_array)/60 + 1, 5))  # X-ticks with intervals of 5 minutes  

    loads_zero_drift = []  
    for i in range(len(load_thruster_array)):     
        load_zero_drift = load_thruster_array[i] - (a_closed*time_load_array[i]**2 + b_closed *time_load_array[i] + c_closed)
        loads_zero_drift.append(load_zero_drift)    
       
    #Filtering
    filtered_loads_zero_drift = filter_data_savgol(loads_zero_drift)
    
    # Plot filtered and unfiltered load data for the current test
    plt.figure()
    plt.plot(time_load_array/60, loads_zero_drift, label='Thrust (Unfiltered)')
    plt.plot(time_load_array/60, filtered_loads_zero_drift, label='Thrust (Filtered)')
    plt.xlabel('Time [min]')
    plt.ylabel('Thrust [mN]')
    # plt.xlim(0, 2500)
    # plt.ylim(-2, 11)
    plt.title(f'{testname}: Thrust with drift subtracted ')
    #plt.ylim(25, 35)
    plt.legend(loc = 'upper right')
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time_load_array)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    
    return filtered_loads_zero_drift #loads_zero_drift

def compute_loads_data_closed(load_thruster_array, time_load_array, valve_open_start_times, valve_open_end_times):
    loads_data_closed = []
    time_loads_closed = []
    index_end_list = []

    for i, (start_time, end_time) in enumerate(zip(valve_open_start_times, valve_open_end_times)):
        index_start = np.abs(time_load_array - start_time).argmin()
        index_end = np.abs(time_load_array - end_time).argmin()
        index_end_list.append(index_end)
        if i == 0:
            loads_data_closed.append(load_thruster_array[:index_start-30])
            time_loads_closed.append(time_load_array[:index_start-30])
        else:
            loads_data_closed.append(load_thruster_array[index_end_list[i-1]+300:index_start-30])
            time_loads_closed.append(time_load_array[index_end_list[i-1]+300:index_start-30])
            

    last_load_closed = load_thruster_array[index_end_list[i]+300:] 
    last_time_loads_closed = time_load_array[index_end_list[i]+300:] 
        
    loads_data_closed.append(last_load_closed)
    time_loads_closed.append(last_time_loads_closed)

    ## Format data from list to usable
    loads_data_closed = np.concatenate(loads_data_closed)
    time_loads_closed = np.concatenate(time_loads_closed)
    
    return loads_data_closed, time_loads_closed

def mean_zero_drift(loads_zero_drift, time_zero_drift):
    # Initialize a variable to store the index where a new test starts
    start_index = 0
    
    # Initialize a list to store the means of each test
    test_means = []
    test_std   = [] 
    
    # Iterate over the cut time loads
    for i in range(1, len(time_zero_drift)):
        # Check if there's at least 200 seconds between the current time and the previous time
        if time_zero_drift[i] - time_zero_drift[i - 1] >= 200:
            # Calculate the mean of the data for the current test
            test_mean = np.mean(loads_zero_drift[start_index:i-1])
            std_deviation = np.std(loads_zero_drift[start_index:i-1])
            # Append the mean to the list of test means
            test_means.append(test_mean)
            test_std.append(std_deviation)
            # Update the start index for the next test
            start_index = i
    
    # Calculate the mean of the data for the last test (from the last start index to the end)
    last_test_mean = np.mean(loads_zero_drift[start_index:])
    last_std_deviation = np.std(loads_zero_drift[start_index:])
    test_means.append(last_test_mean)
    test_std.append(last_std_deviation)
    
    return test_means, test_std

''' Code to aid post-processing'''            
def find_valve_open_times(time_valve_array, valve_array):
    valve_open_start_times = []
    valve_open_end_times = []
    for i in range(len(valve_array) - 1):
        if valve_array[i] == 0 and valve_array[i + 1] >= 5:
            valve_open_start_times.append(time_valve_array[i])
        elif valve_array[i] >= 5 and valve_array[i + 1] == 0:
            valve_open_end_times.append(time_valve_array[i])
    return valve_open_start_times, valve_open_end_times

def compute_cut_indexes(time_array, data_array, start_time, end_time, startup_period=90.0, shutdown_period=3.0, custom_startup_period=None):
    # use custom startup period if provided
    if custom_startup_period is not None:
        startup_period = custom_startup_period    

    # Compute indexes for start and end times
    index_start = np.abs(time_array - start_time).argmin()
    index_end = np.abs(time_array - end_time).argmin()
    
    # Ensure that there is at least `startup_period` seconds of data after start time 
    # and `shutdown_period` seconds before end time
    cut_index_end = index_end - int(shutdown_period / (time_array[1] - time_array[0]))
    cut_index_start = index_start + int(startup_period / (time_array[1] - time_array[0]))
    # Return the cut data directly and compute its mean
    cut_data = data_array[cut_index_start:cut_index_end]
    mean = np.mean(cut_data)
    std_deviation = 3* np.std(cut_data)
    
    return cut_data, mean, std_deviation

def compute_uncertainty(std_deviations):
    """
    Compute uncertainty given an array of standard deviations."""
    # Compute root mean square (RMS) of standard deviations
    RMS_value = np.sqrt(np.sum(np.square(std_deviations)) / len(std_deviations))
    
    # Compute uncertainty using RMS value
    uncertainty = 3 * RMS_value / np.sqrt(len(std_deviations))
    
    return uncertainty

def compute_standard_deviation(values):
    # Step 1: Calculate the mean
    mean = sum(values) / len(values)
    
    # Step 2: Compute the squared differences from the mean
    squared_diffs = [(x - mean) ** 2 for x in values]
    
    # Step 3: Compute the mean of the squared differences
    mean_squared_diffs = sum(squared_diffs) / len(values)
    
    # Step 4: Take the square root of the mean
    standard_deviation = mean_squared_diffs ** 0.5
    
    # Multiply standard deviation by 3 to get 3σ
    three_sigma = 3 * standard_deviation
    
    return three_sigma

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

def process_results(result):
    # testname = 'H2O Test 400°C 1.0 Bar:'
    Time_valve_array = result[0].loc[9, 'x'] #Based on valve open close other charachteristics can be determined
    Valve_array = result[0].loc[9, 'y']
    plot_valveposition(Time_valve_array, Valve_array, testname)

    time_load_array =  result[0].loc[12, 'x']  # Assuming 'x' column represents time
    load_thruster_array = result[0].loc[12, 'y']* (1 / 0.02596344) * (25 / 187.5)  # Assuming 'y' column represents load data
    plot_load(time_load_array, load_thruster_array, testname)

    density = 0.997 # @24°C [g/cm^3] == [mg/uL] NIST 
    time_massflow_array = result[1].loc[:, 'Relative Time[s]'] + 60
    massflow_array = result[1].loc[:, 'Flow [ul/min]']*(density/60) #[uL/min] to [mg/s]
    filtered_massflow_array = filter_data_savgol(massflow_array)
    plot_massflow(time_massflow_array, massflow_array, filtered_massflow_array, testname)

    start_timeM = max(min(time_massflow_array), min(time_load_array))
    end_timeM = min(max(time_massflow_array), max(time_load_array))
    print('START', start_timeM)
    print('END', end_timeM)

    mask_massflow = (time_massflow_array >= start_timeM) & (time_massflow_array <= end_timeM)
    mask_other = (time_load_array >= start_timeM) & (time_load_array <= end_timeM)

    time_massflow_arrayM = time_massflow_array[mask_massflow]
    massflow_data = massflow_array[mask_massflow]

    time_other = time_load_array[mask_other]
    interpolated_massflow = np.interp(time_other, time_massflow_arrayM, massflow_data)
    filtered_massflow_inter = filter_data_savgol(interpolated_massflow)
    plot_massflow(time_other, interpolated_massflow, filtered_massflow_inter, testname)

    time_power_array = result[0].loc[3, 'x']
    power_array = result[0].loc[3, 'y'] + result[0].loc[7, 'y']
    filtered_power_array = filter_data_savgol(power_array)
    plot_power(time_power_array, power_array, filtered_power_array, testname)

    time_pressureTC_array = result[0].loc[13, 'x']
    pressureTC_array = result[0].loc[13, 'y']  #[sccm]
    filtered_pressure_array = filter_data_savgol(pressureTC_array)
    plot_pressure(time_pressureTC_array, pressureTC_array, filtered_pressure_array, testname)
    
    time_temperaturePS_array = result[0].loc[14, 'x']
    temperaturePS_array = result[0].loc[14, 'y']  #[sccm]
    plot_temperature(time_temperaturePS_array, temperaturePS_array, 'at interface', testname)
    
    time_pressurevacuum_array = result[0].loc[15, 'x']
    pressurevacuum_array = result[0].loc[15, 'y']  #
    plot_vacuumpressure(time_pressurevacuum_array, pressurevacuum_array, testname)
    
    Time_temperature_array = result[0].loc[16, 'x']  #
    Temperature1_array =  result[0].loc[16, 'y']  # Temperature thruster
    Temperature2_array =  result[0].loc[17, 'y']  # Temperature bottom propellant line
    Temperature3_array =  result[0].loc[18, 'y']  # Temperature heater 1 (bottom)
    plot_temperature(Time_temperature_array, Temperature1_array, 'Thrust chamber', testname)
    plot_temperature(Time_temperature_array, Temperature2_array, 'Propellant line', testname)
    plot_temperature(Time_temperature_array, Temperature3_array, 'Heater 1 (bottom)', testname)

    # Plot load data with Butterworth filter 
    filtered_loads_data_butterworth = filter_data_butterworth(load_thruster_array)
    
    # Plot load data with Savitzky-Golay filter 
    filtered_loads_data_savgol = filter_data_savgol(load_thruster_array)

    # Plot filtered and unfiltered load data for the current test
    plt.figure()
    plt.plot(time_load_array/60, load_thruster_array, label='Thrust (Unfiltered)')
    #plt.plot(time_load_array/60, filtered_loads_data_butterworth, label='Load (Butterworth Filter)')
    plt.plot(time_load_array/60, filtered_loads_data_savgol, label='Thrust (Filtered)')
    plt.xlabel('Time [min]')
    plt.ylabel('Thrust [mN]')
    plt.title(f'{testname}: Thrust')
    plt.legend(loc = 'upper right')
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time_load_array)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    
    # Initialize a counter for the test number
    test_number = 1
    i = 0
   
    # Initialize lists to store mean loads and standard deviations
    mean_loads, mean_massflows, mean_pressures, mean_temperaturethrusters, mean_temperatureambients, mean_pressureambients, mean_powers      = [], [], [], [], [], [], []
    std_deviations_loads, std_deviations_massflows, std_deviations_pressures, std_deviations_temperaturethrusters, std_deviations_temperatureambients, std_deviations_pressureambients, std_deviations_powers = [], [], [], [], [], [], []
     
    # Determine the time that the valve opens and closes       
    valve_open_start_times, valve_open_end_times = find_valve_open_times(Time_valve_array, Valve_array)

    # Compute loads_data_closed and time_loads_closed
    loads_data_closed, time_loads_closed = compute_loads_data_closed(load_thruster_array, time_load_array, valve_open_start_times, valve_open_end_times)
    
    # Remove random 0 line drift from thrust data
    loads_zero_drift = Remove_drift(load_thruster_array, loads_data_closed, time_load_array, time_loads_closed)
    
    plot_TP(time_load_array, loads_zero_drift, time_pressureTC_array, filtered_pressure_array, testname)
    plot_MP(time_other, filtered_massflow_inter, time_pressureTC_array, filtered_pressure_array, testname)
    plot_MT(time_other, filtered_massflow_inter, time_load_array, loads_zero_drift, testname)
    plot_TT(time_load_array, loads_zero_drift, Time_temperature_array, Temperature1_array, testname)

    # Compute cut data and mean for mass flow, pressure, and power
    for i, (start_time, end_time) in enumerate(zip(valve_open_start_times, valve_open_end_times)):
        cut_thrust_data, mean_thrust, sigma_thrust = compute_cut_indexes(time_load_array, loads_zero_drift, start_time, end_time, custom_startup_period=145.0)
        cut_massflow_data, mean_massflow, sigma_massflow = compute_cut_indexes(time_massflow_array, filtered_massflow_array, start_time, end_time)
        cut_pressure_data, mean_pressure, sigma_pressure = compute_cut_indexes(time_pressureTC_array, pressureTC_array, start_time, end_time)
        cut_temperatethruster_data, mean_temperaturethruster, sigma_temperaturethruster = compute_cut_indexes(Time_temperature_array, Temperature1_array, start_time, end_time)
        cut_temperateambient_data, mean_temperatureambient, sigma_temperatureambient = compute_cut_indexes(Time_temperature_array, Temperature3_array, start_time, end_time)
        cut_pressureambient_data, mean_pressureambient, sigma_pressureambient = compute_cut_indexes(time_pressurevacuum_array, pressurevacuum_array, start_time, end_time)
        cut_power_data, mean_power, sigma_power = compute_cut_indexes(time_power_array, power_array, start_time, end_time)
        print('Cut massflow', cut_massflow_data)
        print('start time', start_time/60)
        print('end time', end_time/60)
        # Calculate the pressure difference between start and end of the cut period
        pressure_difference = cut_pressure_data[-1] - cut_pressure_data[0]    
        
        # Print results
        print(f"H2O-15-300.{test_number}:")
        print(f"H2O-15-300.{test_number} Thrust [mN]: {mean_thrust:.3f} \u00B1 {sigma_thrust:.3f} ({sigma_thrust/mean_thrust*100:.3f})%")
        print(f"H2O-15-300.{test_number} Massflow [mg/s]: {mean_massflow:.3f} \u00B1 {sigma_massflow:.3f}({sigma_massflow/mean_massflow*100:.3f})%")
        print(f"H2O-15-300.{test_number} Pressure [mbar]: {mean_pressure:.3f} \u00B1 {sigma_pressure:.3f}({sigma_pressure/mean_pressure*100:.3f})%")
        print(f"H2O-15-300.{test_number} Temperature of the thruster [C]: {mean_temperaturethruster:.3f} \u00B1 {sigma_temperaturethruster:.3f}({sigma_temperaturethruster/mean_temperaturethruster*100:.3f})%")
        print(f"H2O-15-300.{test_number} Heating power [W]: {mean_power:.1f} \u00B1 {sigma_power:.1f}({sigma_power/mean_power*100:.1f})%")
        print(f"H2O-15-300.{test_number} Ambient temperature [C]: {mean_temperatureambient:.1f} \u00B1 {sigma_temperatureambient:.1f}({sigma_temperatureambient/mean_temperatureambient*100:.1f})%")
        print(f"H2O-15-300.{test_number} Ambient pressure [mbar]: {mean_pressureambient:.1f} \u00B1 {sigma_pressureambient:.1f}({sigma_pressureambient/mean_pressureambient*100:.1f})%")

        # Append mean values and standard deviations 
        mean_loads.append(mean_thrust)
        mean_massflows.append(mean_massflow)
        mean_pressures.append(mean_pressure)
        mean_temperaturethrusters.append(mean_temperaturethruster)
        # mean_tempthruster_Ks.append(mean_tempthruster_K)
        mean_pressureambients.append(mean_pressureambient)
        mean_temperatureambients.append(mean_temperatureambient)
        mean_powers.append(mean_power)
        
        std_deviations_loads.append(sigma_thrust)
        std_deviations_massflows.append(sigma_massflow)
        std_deviations_pressures.append(sigma_pressure)
        std_deviations_temperaturethrusters.append(sigma_temperaturethruster)
        # std_deviations_tempthruster_Ks.append(sigma_tempthruster_K)
        std_deviations_temperatureambients.append(sigma_temperatureambient)
        std_deviations_pressureambients.append(sigma_pressureambient)
        std_deviations_powers.append(sigma_power)
        test_number += 1
     
    # Compute uncertainty from the combined tests       
    uncertainty_load = compute_standard_deviation(mean_loads)
    uncertainty_massflow = compute_standard_deviation(mean_massflows)
    uncertainty_pressure = compute_standard_deviation(mean_pressures)
    uncertainty_temperaturethrusters = compute_standard_deviation(mean_temperaturethrusters)
    # uncertainty_tempthruster_K = compute_standard_deviation(mean_tempthruster_Ks)
    uncertainty_power = compute_standard_deviation(mean_powers)
    uncertainty_pressureambients = compute_standard_deviation(mean_pressureambients)
    uncertainty_temperatureambients = compute_standard_deviation(mean_temperatureambients)
    
    # Compute specific impulse
    g = 9.80665  # Acceleration due to gravity (in m/s^2)
    mean_massflows_kg_s = np.array(mean_massflows)/10**6 #convert mg/s to kg/s
    mean_loads_N = np.array(mean_loads)/1000             #convert mN to N
    specific_impulse = mean_loads_N / (mean_massflows_kg_s * g)   
    
    ##To monitor how much the values change per test    
    RMS_thrust = np.sqrt(np.sum(np.square(mean_loads)) / len(mean_loads))
    RMS_massflow = np.sqrt(np.sum(np.square(mean_massflows)) / len(mean_loads))
    RMS_pressure = np.sqrt(np.sum(np.square(mean_pressures)) / len(mean_loads))
    RMS_temperaturethrusters = np.sqrt(np.sum(np.square(mean_temperaturethrusters)) / len(mean_loads))
    RMS_power = np.sqrt(np.sum(np.square(mean_powers)) / len(mean_loads))
    # RMS_tempthruster_K = np.sqrt(np.sum(np.square(mean_tempthruster_Ks)) / len(mean_tempthruster_Ks))
    RMS_Isp = np.sqrt(np.sum(np.square(specific_impulse)) / len(specific_impulse))
    RMS_ambientpressure = np.sqrt(np.sum(np.square(mean_pressureambients)) / len(mean_loads))
    RMS_ambienttemperature = np.sqrt(np.sum(np.square(mean_temperatureambients)) / len(mean_loads))

    #Error propegation as is listed in chapter analysis
    uncertainty_Isp = np.sqrt((uncertainty_load/RMS_thrust)**2 + (uncertainty_massflow/RMS_massflow)**2) * RMS_Isp
    
    print("Average thrust [mN]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_thrust, uncertainty_load, uncertainty_load/RMS_thrust*100))
    print("Average massflow [mg/s]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_massflow, uncertainty_massflow, uncertainty_massflow/RMS_massflow*100))
    print("Average pressure [mbar]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_pressure, uncertainty_pressure, uncertainty_pressure/RMS_pressure*100))
    print("Average temperature of the thruster [C]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_temperaturethrusters, uncertainty_temperaturethrusters, uncertainty_temperaturethrusters/RMS_temperaturethrusters*100))
    print("Average ambient pressure [mbar]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_ambientpressure, uncertainty_pressureambients, uncertainty_pressureambients/RMS_ambientpressure*100))
    print("Average ambient temperature [C]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_ambienttemperature, uncertainty_temperatureambients, uncertainty_temperatureambients/RMS_ambienttemperature*100))
    print("Average specific impulse [s]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_Isp, uncertainty_Isp, uncertainty_Isp/RMS_Isp*100))
    print("Average power [W]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_power, uncertainty_power, uncertainty_power/RMS_power*100))
    
    print("\n")  # Add a newline for readability between datasets
    Cd, eta, Re, eisp = correction_factor(RMS_massflow, RMS_pressure, RMS_temperaturethrusters, RMS_power, RMS_Isp, RMS_ambientpressure)
    print("\n")  # Add a newline for readability between datasets
    print("Discharge Coefficient [-]: {:.3f}".format(Cd))
    print("Propellant consumption quality (ξ_Isp) [-]: {:.3f}".format(eisp))
    print("Reynolds number [-]: {:.3f}".format(Re))
    print("Heater efficiency [-]: {:.3f}".format(eta))

    # with open("30_08_Water_300_1.0Bar.txt", "w") as text_file:
    with open(f'{testname}.txt', "w") as text_file:
        print("Average thrust [mN]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_thrust, uncertainty_load, uncertainty_load/RMS_thrust*100), file=text_file)
        print("Average massflow [mg/s]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_massflow, uncertainty_massflow, uncertainty_massflow/RMS_massflow*100), file=text_file)
        print("Average pressure [mbar]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_pressure, uncertainty_pressure, uncertainty_pressure/RMS_pressure*100), file=text_file)
        print("Average temperature of the thruster [C]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_temperaturethrusters, uncertainty_temperaturethrusters, uncertainty_temperaturethrusters/RMS_temperaturethrusters*100), file=text_file)
        print("Average ambient pressure [mbar]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_ambientpressure, uncertainty_pressureambients, uncertainty_pressureambients/RMS_ambientpressure*100), file=text_file)
        print("Average ambient temperature [C]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_ambienttemperature, uncertainty_temperatureambients, uncertainty_temperatureambients/RMS_ambienttemperature*100), file=text_file)
        print("Average specific impulse [s]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_Isp, uncertainty_Isp, uncertainty_Isp/RMS_Isp*100), file=text_file)
        print("Average power [W]: {:.3f} \u00B1 {:.3f} ({:.3f}%)".format(RMS_power, uncertainty_power, uncertainty_power/RMS_power*100), file=text_file)
        print("\n", file=text_file)  # Add a newline for readability between datasets
        print("Discharge Coefficient [-]: {:.3f}".format(Cd), file=text_file)
        print("Propellant consumption quality [-]: {:.3f}".format(eisp), file=text_file)
        print("Reynolds number [-]: {:.3f}".format(Re), file=text_file)
        print("Heater efficiency [-]: {:.3f}".format(eta), file=text_file)
    
    # multipage(f'{testname}.pdf')
    plt.show()
    #correction_factor(RMS_massflow, RMS_pressure, RMS_temperaturethrusters, RMS_power, RMS_Isp, RMS_ambientpressure)

result0 = read_tdms(r"Water tests\20240912 H2O_300\135353.tdms")
result1 = pd.read_csv(r"Water tests\20240912 H2O_300\H2O_300.csv",  skiprows=14, delimiter=';')
result1['Flow [ul/min]'] = result1['Flow [ul/min]'].str.replace(',', '.').astype(float)
result1['Relative Time[s]'] = result1['Relative Time[s]'].str.replace('.', '')
result1['Relative Time[s]'] = result1['Relative Time[s]'].str.replace(',', '.').astype(float)
result = [result0, result1]

# testname = "H2O_Test_300°C"
testname = "H2O-15-300A"
print(testname)
print(result)
process_results(result)