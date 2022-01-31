import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

from . import Measurement, Colors, TimeSeries
from common.units import *


"""Plots"""
__plot_baseline = False  # Plots the measured thrust and baseline
__plot_corrected_thrust = False  # Plots the baseline corrected thrust


def run(*measurements: Measurement):
    """Corrects for thrust drift and calculates the produced thrust"""
    if not measurements:
        measurements = Measurement.all

    # Loop through all measurements with type Measurement.THRUST_TEST_N2
    measurement_data = {}
    for m in (m for m in measurements if m.type == Measurement.THRUST_TEST_N2):
        # Create copy of the heater power and filter it
        total_heater_power_filtered = m.total_heater_power.copy()
        total_heater_power_filtered.filter(2)

        # Determine important times in the measurement
        intervals = __find_intervals(m)  # time intervals where the actuator is on, but the valve isn't open

        # Get the mask of the data points that we are going to fit with
        idx = np.zeros((len(m.thrust.time)), dtype=bool)
        for interval in intervals:
            start, end = interval
            idx |= (start + 150 <= m.thrust.time) * (m.thrust.time <= end - 5)

        # Find the baseline with fit function
        fit_func = lambda x, a, b, c: a * x**2 + b*x + c
        popt, pcov = curve_fit(fit_func, m.thrust.time[idx], m.thrust.values[idx])
        popt_ks = popt * np.array([1E6, 1E3, 1])
        fit_str = 'y={:.3f}x$^2${:+.2f}x{:+.2f}'.format(*popt_ks) + '  (x in [ks])'

        # Create a TimeSeries of the baseline
        baseline = TimeSeries(m.thrust.time, fit_func(m.thrust.time, *popt), name=fit_str, unit=m.thrust.unit)

        # Create a copy of the thrust and filter it
        filtered_thrust = m.thrust.copy()
        filtered_thrust.filter(2)

        # Subtract the baseline from the thrust and create a filtered copy
        corrected_thrust = m.thrust - baseline
        corrected_thrust_filtered = corrected_thrust.copy()
        corrected_thrust_filtered.filter(2)

        if __plot_baseline:
            # Plot measured thrust and baseline
            baseline.plot(label=True, c=Colors.red, show=False, zorder=120)
            filtered_thrust.plot(label=True, c=Colors.blue, alpha=0.8, show=False, zorder=110)
            m.thrust.plot(label=True, c=Colors.black, show=True, zorder=100)

        if __plot_corrected_thrust:
            # Plot baseline corrected thrust
            corrected_thrust.plot_with(corrected_thrust_filtered)

        # Gather the data of the thrust periods
        data = []
        for i, interval in enumerate(intervals[1:]):
            # Find the produced thrust
            start, end = interval
            duration = end - start

            thrust_time = start - duration / 2, start - 5
            thrust_avg, thrust_std = corrected_thrust_filtered.avg(*thrust_time)
            pc_avg, pc_std = m.chamber_pressure.avg(*thrust_time)
            pa_avg, pa_std = m.ambient_pressure.avg(*thrust_time)
            tc_avg, tc_std = m.chamber_temperature.avg(*thrust_time)
            mdot_avg, mdot_std = m.mass_flow.avg(*thrust_time)
            power_avg, power_std = total_heater_power_filtered.avg(*thrust_time)

            # Get the average values
            thrust = Force.from_mN(thrust_avg, error=thrust_std * 3)
            pc = Pressure.from_mbar(pc_avg, error=pc_std * 3)
            tc = Temperature.from_celcius(tc_avg, error=tc_std * 3)
            mdot = MassFlow.from_mgps(mdot_avg, error=mdot_std * 3)
            pa = Pressure.from_mbar(pa_avg, error=pa_std * 3)
            pheat = Watt(power_avg, error=power_std * 3)

            data.append({
                'thrust': thrust, 'chamber_pressure': pc, 'chamber_temperature': tc,
                'mass_flow': mdot, 'ambient_pressure': pa, 'heater_power': pheat
            })

        # Save the data of this file into a container
        measurement_data[m.filename] = data

    # Return data of all measurements
    return measurement_data


def __find_intervals(m: Measurement):
    """
    Finds the time intervals where the actuator is on, but the valve isn't open.
    :param m: the measurement to find the intervals in
    :return: [ (start_1, stop_1), (start_2, stop_2), ... ]
    """
    result = []
    start, end = 0, 0
    for t, current in m.actuator_current.iter():
        if start != 0 and current < 0.1 and t > start + 60:
            end = t
            break
        if current > 0.1 and start == 0:
            start = t

    valve_dt = 1 / m.valve.sampling_freq()
    for t, valve in m.valve.iter(start, end):
        if start == 0 and valve < 128:
            # Valve opens
            start = t - (valve_dt / 2)
        else:
            if valve > 128 and start != 0:
                result.append((start, t - (valve_dt / 2)))
                start = 0
    result.append((start, end))

    # Size should be 4 since we do 3 tests
    assert len(result) == 4, f'unexpected number of intervals: {len(result)}'

    return result
