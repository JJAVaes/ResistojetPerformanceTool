from typing import Tuple, List

import numpy as np
from pandas import DataFrame

from data import utils
from data.tdms_reader import read_tdms, Columns, Channels
from . import TimeSeries
from common.units import *
from common import constants
from common.constants import Lever, ActuatorCalibration

Number = Union[float, int]
NumberPair = Tuple[Number, Number]


class Measurement:
    # Static variable containing all measurement instances
    all: List['Measurement'] = []

    # Measurement types
    THRUST_TEST_N2 = 0
    THRUST_TEST_H20 = 1

    def __init__(self, filename: str, type: int, description: Optional[str] = None,
                 constant_mass_flow: Optional[Number] = None,
                 lever_configuration: int = Lever.CONFIGURATION_HUTTEN, **kwargs):

        # Add the instance created to a list containing all measurements
        self.all.append(self)

        self.filename = filename
        self.description = description
        self.type = type
        self.kwargs = kwargs
        self.constant_mass_flow = constant_mass_flow
        self.lever_configuration = lever_configuration

        self.__data_loaded = False

    def __load_data(self):
        """Loads the data from the file and creates the necessary TimeSeries"""
        # Can only call this function once
        if self.__data_loaded:
            return
        self.__data_loaded = True

        # Load the data from the file
        self.data: DataFrame = read_tdms(self.filename)

        # Measurement data
        if self.constant_mass_flow:
            self.mass_flow = TimeSeries(time=np.array(self.get_time_limits()),
                                        values=np.array([self.constant_mass_flow, self.constant_mass_flow]),
                                        name="Mass flow", unit='mg/s')
        else:
            self.mass_flow = utils.get_time_series(self.data, Channels.FEED_SYSTEM_MASS_FLOW, name="Mass flow")
            self.mass_flow = (self.mass_flow * constants.sccm_to_mgps).astype(unit="mg/s")

        self.pendulum_distance = utils.get_time_series(self.data, Channels.PENDULUM_DISTANCE)
        self.setpoint = utils.get_time_series(self.data, Channels.DISTANCE_SETPOINT)
        self.chamber_pressure = utils.get_time_series(self.data, Channels.PRESSURE_AT_INTERFACE, name="Chamber Pressure")
        self.ambient_pressure = utils.get_time_series(self.data, Channels.VACUUM_CHAMBER_PRESSURE, name="Vacuum Chamber Pressure")
        self.temperature_at_interface = utils.get_time_series(self.data, Channels.TEMPERATURE_AT_INTERFACE, name="Temperature at interface")
        self.actuator_current = utils.get_time_series(self.data, Channels.ACTUATOR_CURRENT)
        self.actuator_voltage = utils.get_time_series(self.data, Channels.ACTUATOR_VOLTAGE)
        self.valve = utils.get_time_series(self.data, Channels.VALVE_STATE)
        self.heater_1_power = utils.get_time_series(self.data, Channels.HEATER_1_POWER)
        self.heater_2_power = utils.get_time_series(self.data, Channels.HEATER_2_POWER)
        self.heater_1_current = utils.get_time_series(self.data, Channels.HEATER_1_CURRENT)
        self.heater_2_current = utils.get_time_series(self.data, Channels.HEATER_2_CURRENT)
        self.heater_1_resistance = utils.get_time_series(self.data, Channels.HEATER_1_RESISTANCE)
        self.heater_2_resistance = utils.get_time_series(self.data, Channels.HEATER_2_RESISTANCE)
        self.heater_1_voltage = utils.get_time_series(self.data, Channels.HEATER_1_VOLTAGE)
        self.heater_2_voltage = utils.get_time_series(self.data, Channels.HEATER_2_VOLTAGE)

        self.chamber_temperature = utils.get_time_series(self.data, Channels.TC_1_TEMPERATURE, name="Temperature")
        self.tc_2_temp = utils.get_time_series(self.data, Channels.TC_2_TEMPERATURE, name="Temperature")
        self.tc_3_temp = utils.get_time_series(self.data, Channels.TC_3_TEMPERATURE, name="Temperature")
        self.tc_4_temp = utils.get_time_series(self.data, Channels.TC_4_TEMPERATURE, name="Temperature")

        if self.tc_2_temp and self.tc_3_temp and self.tc_4_temp:  # Ensure all exist
            self.temperature_tube_bottom, self.temperature_tube_middle, self.temperature_tube_top = \
                sorted([self.tc_2_temp, self.tc_3_temp, self.tc_4_temp], key=lambda m: m.avg())  # type: TimeSeries

        # Calculate thrust
        actuator_calibration = ActuatorCalibration(Lever(self.lever_configuration))
        self.thrust = TimeSeries(self.actuator_current.time, self.actuator_current.values * actuator_calibration.get_factor(), name="Thrust", unit="mN")

        # Calculate total heater power/current (total heater voltage should be defined by P/I, does not really exist)
        self.total_heater_power = (self.heater_1_power + self.heater_2_power).astype("Total heater power")

    def __getattr__(self, attr):
        """
        Only load the data from the measurement if we actually want to access it.
        """
        # Load data from disk and process
        if attr not in self.__dict__:
            self.__load_data()

        # Try to access the value again. Raises AttributeError if the attribute is still not present.
        return super(Measurement, self).__getattribute__(attr)

    def get_time_limits(self):
        time = self.data.loc[0][Columns.X]
        return time[0], time[-1]
