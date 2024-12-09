"""
@author: L.C.M. Cramer
@editor: J.J.A. Vaes

Created on 15-02-2024 12:01:35
Edited on 08-12-2024 16:43:00 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
'''To plot all usefull graphs'''

def plot_valveposition(time,valveposition, testname):
    plt.figure()
    plt.plot(time/60, valveposition)
    plt.xlabel('Time [min]')
    plt.ylabel('Valve position')
    plt.ylim()
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.title(f'{testname}: Valve position')  

def plot_load(time,load, testname):
    plt.figure()
    plt.plot(time/60, load)
    plt.xlabel('Time [min]')
    plt.ylabel('Thrust [mN]')
    plt.ylim() 
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.title(f"{testname}: Thrust")  
    
def plot_massflow(time, massflow, massflow_filtered, testname):
    plt.figure()
    plt.plot(time/60, massflow, label='Massflow (Unfiltered)')
    plt.plot(time/60, massflow_filtered, label='Massflow (Filtered)')
    plt.xlabel('Time [min]')
    plt.ylabel('Massflow [mg/s]')
    plt.ylim() 
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.legend(loc = 'upper right')
    plt.title(f"{testname}: Massflow")  
    
def plot_pressure(time, pressure, pressure_filtered, testname):
    plt.figure()
    plt.plot(time/60, pressure, label='Pressure at interface (Unfiltered)')
    plt.plot(time/60, pressure_filtered, label='Pressure at interface (Filtered)')
    plt.xlabel('Time [min]')
    plt.ylabel('Pressure [mbar]')
    plt.ylim() 
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.legend(loc = 'upper right')
    plt.title(f"{testname}: Pressure at interface")  
    
def plot_vacuumpressure(time, pressure, testname):
    plt.figure()
    plt.plot(time/60, pressure)
    plt.xlabel('Time [min]')
    plt.ylabel('Pressure [mbar]')
    plt.ylim() 
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.title(f"{testname}: Ambient pressure")  
    
def plot_temperature(time,temperature, label, testname):
    plt.figure()
    #plt.figure(figsize=(3,6))
    plt.plot(time/60, temperature)
    plt.xlabel('Time [min]')
    plt.ylabel('Temperature [°C]')
    plt.ylim() 
    #plt.xlim(-1,11)
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.title(f"{testname}: Temperature {label}")  
    
def plot_power(time, power, power_filtered, testname):
    plt.figure()
    plt.plot(time/60, power, label='Power (Unfiltered)')
    plt.plot(time/60, power_filtered, label='Power (Filtered)')
    plt.xlabel('Time [min]')
    plt.ylabel('Power [W]')
    plt.ylim() 
    plt.grid(True)
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    plt.legend(loc = 'upper right')
    plt.title(f"{testname}: Heater power")  

def plot_TP(time, thrust, time2,  pressure, testname): 
    fig, ax1 = plt.subplots()  # Create a figure and primary axis
    
    # Plot on the first axis (Thrust)
    lns1 = ax1.plot(time / 60, thrust, 'tab:green', label='Thrust (Green)')
    ax1.set_xlabel('Time [min]')
    ax1.set_ylim(-2, 10)
    ax1.set_ylabel('Thrust [mN]')
    ax1.grid(True)  # Enable grid for the first axis
    ax1.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    
    # Create the twin axis (Pressure)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(time2 / 60, pressure, 'tab:orange', linestyle='--', label='Pressure (Orange)')
    ax2.set_ylim(0, 1600)
    ax2.set_ylabel('Pressure [mbar]')
    ax2.grid(False)  # Enable grid for the second axis
    
    # Combine both legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Set the title
    plt.title(f"{testname}: VLM thrust and pressure")

def plot_MP(time, massflow_filtered, time2,  pressure, testname): 
    fig, ax1 = plt.subplots()  # Create a figure and primary axis

    # Plot on the first axis (Thrust)
    lns1 = ax1.plot(time / 60, massflow_filtered, label='Massflow (Blue)')
    ax1.set_xlabel('Time [min]')
    ax1.set_ylim(0, 25)
    ax1.set_ylabel('Massflow [mg/s]')
    ax1.grid(True)  # Enable grid for the first axis
    ax1.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    
    # Create the twin axis (Pressure)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(time2 / 60, pressure, 'tab:orange', linestyle='--', label='Pressure (Orange)')
    ax2.set_ylim(0, 1600)    
    ax2.set_ylabel('Pressure [mbar]')
    ax2.grid(False)  # Enable grid for the second axis
    
    # Combine both legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Set the title
    plt.title(f"{testname}: VLM massflow and pressure")

def plot_MT(time, massflow_filtered, time2,  thrust, testname): 
    fig, ax1 = plt.subplots()  # Create a figure and primary axis

    # Plot on the first axis (Thrust)
    lns1 = ax1.plot(time / 60, massflow_filtered, label='Massflow (Blue)')
    ax1.set_xlabel('Time [min]')
    ax1.set_ylim(0, 25)
    ax1.set_ylabel('Massflow [mg/s]')
    ax1.grid(True)  # Enable grid for the first axis
    ax1.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes

    # Create the twin axis (Pressure)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(time2 / 60, thrust, 'tab:green', linestyle='--', label='Thrust (Green)')
    ax2.set_ylim(-2, 10)
    ax2.set_ylabel('Thrust [mN]')
    ax2.grid(False)  # Enable grid for the second axis
    
    # Combine both legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Set the title
    plt.title(f"{testname}: VLM massflow and thrust")
    
def plot_TT(time, thrust, time2,  temperature, testname): 
    fig, ax1 = plt.subplots()  # Create a figure and primary axis
    
    # Plot on the first axis (Thrust)
    lns1 = ax1.plot(time / 60, thrust, 'tab:green', label='Thrust (Green)')
    ax1.set_xlabel('Time [min]')
    ax1.set_ylim(-2, 10)
    ax1.set_ylabel('Thrust [mN]')
    ax1.grid(True)  # Enable grid for the first axis
    ax1.set_xticks(np.arange(0, max(time)/60 + 1, 5))  # X-ticks with intervals of 5 minutes
    
    # Create the twin axis (Pressure)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(time2 / 60, temperature, 'tab:red', linestyle='--', label='Temperature (Red)')
    ax2.set_ylim(0, 410)
    ax2.set_ylabel('Temperature [°C]')
    ax2.grid(False)  # Enable grid for the second axis
    
    # Combine both legends
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Set the title
    plt.title(f"{testname}: VLM thrust and temperature")

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
