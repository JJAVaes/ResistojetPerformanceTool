"""
@author: L.C.M. Cramer
@editor: J.J.A. Vaes

Created on 15-02-2024 12:01:35
Edited on 08-12-2024 16:43:00 
"""

import csv
import re
from matplotlib import pyplot as plt
from nptdms import TdmsFile
import numpy as np
from pandas import DataFrame, to_datetime

#Seems to work with some help of ChatGPT corrected channels to name
def info_from_name(name):
    """
    Gets title, quantity and unit from name (if they are in there)
    Acceptable formats:
    <title> <{quantity}> <[unit]>
    <title> <[unit]>
    <title>
    Args:
        title (string): title string from labview data file
    Returns:
        tuple: containing:
            string: title (full name string of other items not found)
            string: quantity ('?' if not found)
            string: unit ('?' if not found)
    """
    quantity_match = re.search('(?<={).*(?=})', name)
    unit_match = re.search('(?<=\[).*(?=\])', name)
    if quantity_match is None and unit_match is None:
        title = name
    elif quantity_match is None and unit_match is not None:
        title = re.search('^.*(?= \[)', name).group(0)
    else:
        title = re.search('^.*(?= {)', name).group(0)
    quantity = quantity_match.group(0) if quantity_match is not None else '?'
    unit = unit_match.group(0) if unit_match is not None else '?'
    return title, quantity, unit

def read_lvm(file_path):
    """
    Reads LVM measurement file from labview data acquisition system in SSE cleanroom
    formatted with a single header, each channel name may contain data according to the format
    specified in info_from_name() for title quantity and unit of the data
    The first column of data should be time [s] since VI start
    Args:
        file_path (string): path to file to be read (relative or absolute)
    Returns:
        Pandas DataFrame: with rows for each channel and the following columns:
            x_quantity (string): physical quantity on x-axis
            x_unit (string): unit of values on x-axis
            title (string): title given to y quantity of data
            y_unit (string): unit of values on y-axis
            y_quantity (string): physical quantity on y-axis
            x (array): x data
            y (array): y data
    """
    with open(file_path, 'r') as data_file:
        header_ended = False
        csv_reader = csv.reader(data_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if not header_ended:
                if row[0] == 'Channels': # find number of data channels and initialize array
                    n_channels = int(row[1])
                    data = [dict() for _ in range(n_channels)]
                elif row[0] == 'X_Dimension': # physical quantity in X-vector of each channel
                    for j in range(n_channels):
                        data[j]['x_quantity'] = row[j+1]
                        data[j]['x_unit'] = 's'
                elif row[0] == 'X_Value': # channel titles and start of actual data
                    for j in range(n_channels):
                        title, quantity, unit = info_from_name(name=row[j+1])
                        data[j]['title']      = title
                        data[j]['y_quantity'] = quantity
                        data[j]['y_unit']     = unit
                    header_ended = True
                    i_start = i+1 # used to check if the first row of data is being read
            else:
                if i == i_start: # first row of data
                    for j in range(n_channels): # initialize lists
                        data[j]['x'] = [0.]
                        data[j]['y'] = [float(row[j+1])]
                else: # all subsequent data rows
                    for j in range(n_channels): # add to lists
                        data[j]['x'] += [float(row[0])]
                        data[j]['y'] += [float(row[j+1])]
        for j in range(n_channels): # convert x and y data to numpy arrays
            data[j]['x'] = np.array(data[j]['x'])
            data[j]['y'] = np.array(data[j]['y'])
    return DataFrame(data)

def read_tdms(file_path):
     """
     Reads TDMS measurement file from labview data acquisition system in SSE cleanroom
     Format assumes one group per data acquisition frequency, each with at least one channel for
     time [s] and one or more channels for associated data.
     Channel names are formatted according to the format
     specified in info_from_name() for title quantity and unit of the data
     Args:
         file_path (string): path to file to be read (relative or absolute)
     Returns:
         Pandas DataFrame: with rows for each channel and the following columns:
             x_quantity (string): physical quantity on x-axis
             x_unit (string): unit of values on x-axis
             title (string): title given to y quantity of data
             y_unit (string): unit of values on y-axis
             y_quantity (string): physical quantity on y-axis
             x (array): x data
             y (array): y data
     """
           
     tdms_file = TdmsFile(file_path)
     data = []
     for group in tdms_file.groups():
        channel_indices = []
        group_x_data = None
        group_name = group.name
        # look for time data first
        for i, channel_obj in enumerate(group.channels()):
            channel_name = channel_obj.name
            if channel_name == 'time':
                group_x_data = channel_obj.data
            else:
                channel_indices += [i] # non-time channel found, save for later use
         # if time data was found in group, loop over non-time channels
        if group_x_data is not None:
             for ind in channel_indices:
                 channel_obj = group.channels()[ind]
                 data += [{}] # add new data element to total list
                 data[-1]['x_quantity'] = 'Time' # always assume time in seconds
                 data[-1]['x_unit']     = 's'
                 title, quantity, unit = info_from_name(channel_obj.name)
                 data[-1]['title']      = title
                 data[-1]['y_quantity'] = quantity
                 data[-1]['y_unit']     = unit
                 data[-1]['x'] = group_x_data
                 data[-1]['y'] = channel_obj.data
        else:
             raise RuntimeError('time channel not found in group')
     return DataFrame(data)

def plot_data(df):
         #plots can be closed by hand 
         for i, row in df.iterrows():
             x = row['x']
             y = row['y']
             x_quantity = row['x_quantity']
             x_unit = row['x_unit']
             y_quantity = row['y_quantity']
             y_unit = row['y_unit']
             title = row['title']
            
             plt.figure()
             #plt.scatter(x, y, label='Data Points', color='blue', marker = ',')  # Use scatter instead of plot
             plt.plot(x, y)
             plt.xlabel(f'{x_quantity} ({x_unit})')
             plt.ylabel(f'{y_quantity} ({y_unit})')
             plt.title(title)
             #plt.show()