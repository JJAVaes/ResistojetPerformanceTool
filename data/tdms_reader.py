"""
From Huib Versteeg. Github link:
    https://github.com/spacee11/thesis_code/blob/master/data_analysis/labview_parsers.py
"""

import glob
import os
import re
from nptdms import TdmsFile
from pandas import DataFrame


# DataFrame columns
class Columns:
    TITLE = 'title'
    X = 'x'
    Y = 'y'
    X_QUANTITY = 'x_quantity'
    Y_QUANTITY = 'y_quantity'
    X_UNIT = 'x_unit'
    Y_UNIT = 'y_unit'


# DataFrame channels
class Channels:
    ACTUATOR_CURRENT = 'Actuator current'
    ACTUATOR_CURRENT_SETPOINT = 'Actuator current setpoint'
    ACTUATOR_VOLTAGE = 'Actuator voltage'
    DISTANCE_SETPOINT = 'Distance setpoint'
    FEED_SYSTEM_MASS_FLOW = 'Feed system mass flow'
    HEATER_1_CURRENT = 'Heater 1 Current'
    HEATER_1_POWER = 'Heater 1 Power'
    HEATER_1_RESISTANCE = 'Heater 1 Resistance'
    HEATER_1_VOLTAGE = 'Heater 1 Voltage'
    HEATER_1_VOLTAGE_SETPOINT = 'Heater 1 voltage setpoint'
    HEATER_2_CURRENT = 'Heater 2 Current'
    HEATER_2_POWER = 'Heater 2 Power'
    HEATER_2_RESISTANCE = 'Heater 2 Resistance'
    HEATER_2_VOLTAGE = 'Heater 2 Voltage'
    HEATER_2_VOLTAGE_SETPOINT = 'Heater 2 voltage setpoint'
    PENDULUM_DISTANCE = 'Pendulum distance'
    PRESSURE_AT_INTERFACE = 'Pressure at interface'
    TC_1_TEMPERATURE = 'TC-1 Temperature'
    TC_2_TEMPERATURE = 'TC-2 Temperature'
    TC_3_TEMPERATURE = 'TC-3 Temperature'
    TC_4_TEMPERATURE = 'TC-4 Temperature'
    TEMPERATURE_AT_INTERFACE = 'Temperature at interface'
    VACUUM_CHAMBER_PRESSURE = 'Vacuum chamber pressure'
    VALVE_STATE = 'Valve state'


def read_tdms(file_name: str) -> DataFrame:
    """
    Reads TDMS measurement file from labview data acquisition system in SSE cleanroom
    Format assumes one group per data acquisition frequency, each with at least one channel for
    time [s] and one or more channels for associated data.
    Channel names are formatted according to the format
    specified in info_from_name() for title quantity and unit of the data
    Args:
        file_name (string): path to file to be read (relative or absolute)
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
    if not os.path.exists(file_name):
        file_name = _find_file_path(file_name)

    print('tdms.read_tdms :', 'Using ' + file_name)
    tdms_file = TdmsFile(file_name)
    data = []
    for group in tdms_file.groups():
        channel_indices = []
        group_x_data = None
        # look for time data first
        for i, channel_obj in enumerate(group.channels()):
            channel_name = channel_obj.name
            if channel_name == 'time':
                group_x_data = channel_obj.data
            else:
                channel_indices += [i]  # non-time channel found, save for later use
        # if time data was found in group, loop over non-time channels
        if group_x_data is not None:
            for ind in channel_indices:
                channel_obj = group.channels()[ind]
                data += [{}]  # add new data element to total list
                data[-1][Columns.X_QUANTITY] = 'Time'  # always assume time in seconds
                data[-1][Columns.X_UNIT] = 's'
                title, quantity, unit = _info_from_name(channel_obj.name)
                data[-1][Columns.TITLE] = title
                data[-1][Columns.Y_QUANTITY] = quantity
                data[-1][Columns.Y_UNIT] = unit
                data[-1][Columns.X] = group_x_data
                data[-1][Columns.Y] = channel_obj.data
        else:
            raise RuntimeError('time channel not found in group')
    # data_df = DataFrame(data)
    return DataFrame(data).sort_values(by=[Columns.TITLE]).reset_index(drop=True)


def _find_file_path(name):
    """
    Finds the file path in the data folder
    """
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    files = glob.iglob(data_path + '/**/*.*', recursive=True)
    for file_path in files:
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        if file_name == name or file_name_no_ext == name:
            return file_path
    raise ValueError(f'No file named {name}')


def _info_from_name(name):
    """
    Gets title, quantity and unit from name (if they are in there)
    Acceptable formats:
    <title> <{quantity}> <[unit]>
    <title> <[unit]>
    <title>
    Args:
        name (string): title string from labview data file
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
