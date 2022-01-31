from typing import List, Tuple, Iterator, Callable, get_args

import numpy as np
from matplotlib import pyplot as plt, ticker

from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.stats import linregress

from common.units import *

from analysis import Colors

Number = Union[float, int]
NumberPair = Tuple[Number, Number]
Array = Union[List, Tuple, np.array]
Func = Callable[[np.ndarray, np.ndarray], np.ndarray]  # [time, value] => value


class TimeSeries:
    """
    A TimeSeries represents a unit that is measured over time.
    """

    def __init__(self, time: Array, values: Array, name: str, unit: str = "-", is_filtered: bool = False):
        self.name = name
        self.unit = unit
        self.values = np.array(values)
        self.time = np.array(time)
        self.is_filtered = is_filtered

    def __call__(self, time: Number, derivative=False):
        """Interpolate the time series at the given time"""
        if self.interpolator_needs_update:
            self.interpolator = interpolate.CubicSpline(self.time, self.values)
            self.interpolator_needs_update = False
        if time < self.start() or self.end() < time:
            raise ValueError(f"Requested time {time} out of range [{self.start()},{self.end()}]")
        if derivative:
            return self.interpolator(time, 1)
        return self.interpolator(time)

    def __setattr__(self, key, value):
        """Detect time or values changes to mark the interpolator to need an update"""
        if key in ['time', 'values']:
            self.interpolator_needs_update = True
        super().__setattr__(key, value)

    def __add__(self, other: Union['TimeSeries', Number]):
        """Add TimeSeries to each other. Interpolate other at every measurement time of self."""
        if not isinstance(other, get_args(Union[TimeSeries, Number])):
            raise TypeError(f"Can't add type of {type(other)} to TimeSeries.")
        if isinstance(other, get_args(Number)):
            return TimeSeries(self.time, self.values + other, self.name, self.unit, self.is_filtered)
        if self.unit != other.unit:
            raise TypeError(f"TimeSeries do not have equal units: {self.unit} and {other.unit}.")

        # Get overlapping measurement time
        start_time = max(self.start(), other.start())
        end_time = min(self.end(), other.end())

        if end_time <= start_time:
            raise ValueError(f"TimeSeries do not overlap in time: {self.limits()} and {other.limits()}")

        # Lists to hold the new data
        new_time, new_values = [], []

        # Add TimeSeries together by interpolating other TimeSeries.
        for time, value in self.iter(start_time, end_time):
            new_time.append(time)
            new_values.append(value + other(time))

        return TimeSeries(new_time, new_values, self.name, self.unit, self.is_filtered or other.is_filtered)

    def __radd__(self, other):
        """Right add is the same as left add"""
        return other.__add__(self)

    def __sub__(self, other):
        """Subtract TimeSeries from eachother"""
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return other.__add__(-1 * self)

    def __mul__(self, other: Union['TimeSeries', Number]):
        """Multiply with number or other TimeSeries"""
        if not isinstance(other, get_args(Union[TimeSeries, Number])):
            raise TypeError(f"Can't add type of {type(other)} to TimeSeries.")
        if isinstance(other, get_args(Number)):
            return TimeSeries(self.time, self.values * other, self.name, self.unit, self.is_filtered)

        # Other is of type TimeSeries
        new_time, new_values = [], []
        overlap = max(self.start(), other.start()), min(self.end(), other.end())
        for t, val in self.iter():
            if overlap[0] <= t <= overlap[1]:
                new_values.append(val * other(t))
                new_time.append(t)
        return TimeSeries(time=new_time, values=new_values, name=f"{self.name} {other.name}",
                          unit=f"{self.unit} * {other.unit}", is_filtered=self.is_filtered or other.is_filtered)

    def __rmul__(self, other):
        """Right multiply is the same as left multiply"""
        return self.__mul__(other)

    def __truediv__(self, other: Union['TimeSeries', Number]):
        """Divide with number or other TimeSeries"""
        if not isinstance(other, get_args(Union[TimeSeries, Number])):
            raise TypeError(f"Can't add type of {type(other)} to TimeSeries.")
        if isinstance(other, get_args(Number)):
            return TimeSeries(self.time, self.values / other, self.name, self.unit, self.is_filtered)

        # Other is of type TimeSeries
        new_time, new_values = [], []
        overlap = max(self.start(), other.start()), min(self.end(), other.end())
        for t, val in self.iter():
            if overlap[0] <= t <= overlap[1]:
                new_values.append(val / other(t))
                new_time.append(t)
        return TimeSeries(time=new_time, values=new_values, name=f"{self.name} per {other.name}",
                          unit=f"{self.unit} / {other.unit}", is_filtered=self.is_filtered or other.is_filtered)

    def __rtruediv__(self, other: Number):
        if not isinstance(other, get_args(Number)):
            raise TypeError(f"Can't divide type of {type(other)} by TimeSeries.")

        # Other is of type TimeSeries
        new_values = []
        for t, val in self.iter():
            new_values.append(other / val)
        return TimeSeries(time=self.time, values=new_values, name=f"{self.name}-1$",
                          unit=f"{self.unit}-1", is_filtered=self.is_filtered)

    def clamp(self, min: Number, max: Number):
        """Removes the vales that are outside the given min and max"""
        idx = (min < self.values) * (self.values < max)
        self.values = self.values[idx]
        self.time = self.time[idx]

    def copy(self):
        """Returns a copy of the TimeSeries"""
        return self * 1

    def astype(self, name: str = None, unit:str = None) -> 'TimeSeries':
        """Sets a new name and/or unit"""
        if name:
            self.name = name
        if unit:
            self.unit = unit
        return self

    def apply(self, function: Func):
        """
        Applies a function on the values. The given function must be of one of the following types:
        def func(time, value) -> value
        lambda time, value: value

        The function is modifies the values of the TimeSeries object.
        """
        self.values = function(self.time, self.values)

    def iter(self, start: Optional[Number] = None, end: Optional[Number] = None) -> Iterator:
        """Returns an iterator returning (time, value) combinations in chronological order"""
        start, end = self._get_valid_limits(start, end)
        idx = (self.time >= start) * (self.time <= end)
        return zip(self.time[idx], self.values[idx])

    def dt(self, time: Number, time_range: Number = 1):
        """Calculates the derivative at the given time"""
        idx = (time - time_range/2 <= self.time) * (self.time <= time + time_range/2)
        slope, intercept, r_value, p_value, std_err = linregress(self.time[idx], self.values[idx])
        return slope, std_err
        # return self(time, derivative=True)

    def start(self) -> Number:
        """Returns the first measurement time"""
        return self.time[0]

    def end(self) -> Number:
        """Returns the last measurement time"""
        return self.time[-1]

    def limits(self) -> NumberPair:
        """Returns the first and last measurement time"""
        return self.start(), self.end()

    def range(self, start: Optional[Number] = None, end: Optional[Number] = None) -> NumberPair:
        """Returns the range, [y_min, y_max], of the given domain"""
        start, end = self._get_valid_limits(start, end)
        idx = (self.time >= start) * (self.time <= end)
        return min(self.values[idx]), max(self.values[idx])

    def filter(self, filter_window_seconds: Number):
        """Filters the values so that the result is smoother"""
        total_samples = self.sampling_freq() * filter_window_seconds
        samples_round_to_odd = np.ceil(total_samples) // 2 * 2 + 1
        self.values = savgol_filter(self.values, window_length=int(samples_round_to_odd), polyorder=1)
        self.is_filtered = True

    def sampling_freq(self) -> float:
        """Returns the sampling frequency in Hz. Assumes the sampling frequency is constant"""
        return 1 / float(np.median(np.diff(self.time)))

    def avg(self, start: Optional[Number] = None, end: Optional[Number] = None) -> [Number, Number]:
        """Calculates the average and standard deviation values between the two given values"""
        start, end = self._get_valid_limits(start, end)

        idx = (self.time >= start) * (self.time <= end)
        vals = self.values[idx]
        return np.mean(vals), np.std(vals)

    def confidence_interval(self, start: Optional[Number] = None, end: Optional[Number] = None,
                            z=None, confidence=None) -> [Number, Number]:
        """
        Calculate the confidence interval at the given range and confidence

        CI = x_mean +- z * std_dev / sqrt(sample_size), where z = confidence level value (number of std deviations)

        Arguments:
            start: time at which to start calculating the confidence interval
            end: time at which to stop calculating the confidence interval
            z: number of standard deviations to calculate the interval of
            confidence: [0,1] (if z is None) confidence interval =>
                        0.95 is almost equal to z=2, 99.7 almost equal to z=3
        """
        start, end = self._get_valid_limits(start, end)

        # Can't both be None
        if z is None and confidence is None:
            raise ValueError("Define either 'z' or 'confidence'.")

        avg, std_dev = self.avg(start, end)
        if z is None:
            z = NormalDist().inv_cdf((1 + confidence) / 2.)  # Confidence level value

        sample_size = np.sum((self.time >= start) * (self.time <= end))

        interval = z * std_dev / (np.sqrt(sample_size))
        return avg, interval

    def cut(self, *interval: NumberPair, keep=True):
        """
        Returns a new TimeSeries cut at the given intervals. Does not recalculate the time.

        TimeSeries.cut((3, 5), (8, 10)) => keep data between 3 and 5 seconds, and between 8 and 10 seconds.

        Args:
            keep: True, keep the data in the given intervals (default)
                  False, keep the data outside the given intervals
        """
        if len(interval) == 0:
            return ValueError("At least one interval must be given")

        mask = np.zeros(len(self.time), dtype="bool")

        for start, end in interval:
            mask |= (start <= self.time) * (self.time <= end)

        if not keep:
            mask = ~mask  # Invert if we are meant to remove the intervals from the data

        return TimeSeries(self.time[mask], self.values[mask], self.name, self.unit, self.is_filtered)

    def plot(self, title: str = None, xlim: Optional[NumberPair] = None, ylim: Optional[NumberPair] = None, log: bool = False,
             x_ticks: List[Number] = None, show=True, label: Union[bool, str] = False, c=Colors.black, **kwargs) -> None:
        """Plots the TimeSeries and shows it"""
        plt.grid(b=True, axis='y', zorder=0)
        plt.xlabel("Time [s]")
        plt.ylabel(f"{self.name} [{self.unit}]")

        if type(label) == bool and label:  # label=True defaults to TimeSeries name
            label = self.name
            if self.is_filtered:
                label += ' (filtered)'

        # Add arguments to kwargs
        kwargs['c'] = c
        kwargs['label'] = label

        plt.plot(self.time, self.values, **kwargs)
        plt.tight_layout(rect=(0, 0, 1, 0.92))

        if label:
            plt.legend()
        if x_ticks:
            plt.xticks(x_ticks)
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)
        if title:
            plt.title(title)
        if log:
            plt.yscale('log')
        if show:
            plt.show()

    def plot_with(self, other: 'TimeSeries', title: str = None, limits: NumberPair = None,
                  combine_y_axis: bool = None, ylim1: NumberPair = None, ylim2: NumberPair = None,
                  xlim: NumberPair = None, log1: bool = False, log2: bool = False, include_zero1: bool = False,
                  include_zero2: bool = False, c1: str = Colors.black, c2: str = Colors.blue, alpha1: float = 1,
                  alpha2: float = 0.8, x_ticks: List[Number] = None, show=True, legend_loc: str = 'best'):
        """Plots two TimeSeries in a single graph."""
        fig, ax1 = plt.gcf(), plt.gca()

        ax1.set_xlabel('Time [s]')
        if not ax1.get_ylabel():
            ax1.set_ylabel(f"{self.name} [{self.unit}]")

        label1 = self.name
        if self.is_filtered:
            label1 += ' (filtered)'

        ax1.plot(self.time, self.values, color=c1, label=label1, alpha=alpha1)
        ax1.tick_params(axis='y', labelcolor=Colors.black)

        # If the units of the timeseries are the same, default to combine y axis
        if combine_y_axis is None:
            combine_y_axis = self.unit == other.unit

        if combine_y_axis:
            ax2 = ax1
        else:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel(f"{other.name} [{other.unit}]")
            ax2.tick_params(axis='y', labelcolor=Colors.black)

        label2 = other.name
        if other.is_filtered:
            label2 += ' (filtered)'

        ax2.plot(other.time, other.values, color=c2, label=label2, alpha=alpha2)

        # Set limits
        if xlim:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)

        for ax, lim, incl in ((ax1, ylim1, include_zero1), (ax2, ylim2, include_zero2)):
            # Set limit if given
            if lim:
                ax.set_ylim(lim)
            # Include zero in the graph
            else:
                if incl:
                    min_val, max_val = ax.get_ylim()
                    if math.copysign(1, min_val) == math.copysign(1, max_val):
                        size = max_val - min_val
                        if min_val > 0:
                            ax.set_ylim(min(0, min_val - (size * 0.1)), max_val * 1.1)
                        else:
                            ax.set_ylim(min_val * 1.1, max(0, max_val + size * 0.1))

        # Set log scale
        if log1:
            ax1.set_yscale('log')
        if log2:
            ax2.set_yscale('log')

        # Align ticks of the two y axis
        if not log1 and not log2 and not combine_y_axis:
            ax1_ticks = ax1.get_yticks()
            ax2_ticks = ax2.get_yticks()

            ax1_size = ax1_ticks[-1] - ax1_ticks[0]
            ax2_size = ax2_ticks[-1] - ax2_ticks[0]

            f = lambda y: ax2_ticks[0] + (y - ax1_ticks[0]) / ax1_size * ax2_size
            ax2.set_ylim(f(ax1.get_ylim()))

            ticks = f(ax1_ticks)
            ax2.yaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax1.grid(b=True, axis='y', zorder=0)
        # Don't align ticks when one of them is log scale
        elif log1:
            ax2.grid(b=True, axis='y', zorder=0)
        elif log2:
            ax1.grid(b=True, axis='y', zorder=0)
        else:
            plt.grid(b=True, axis='y', zorder=0)

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        # Get labels from both plots and put them in one legend
        lines, labels = ax1.get_legend_handles_labels()
        if combine_y_axis:
            ax1.legend(lines, labels, loc=legend_loc)
        else:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc=legend_loc)

        if x_ticks:
            plt.xticks(x_ticks)
        if limits:
            plt.xlim(*limits)
        if title:
            plt.title(title)
        if show:
            plt.show()
        else:
            # Continue with the figure, set ax1 as the axis to continue on
            plt.sca(ax1)

    def _get_valid_limits(self, start: Optional[Number] = None, end: Optional[Number] = None):
        """
        Get the start and end time that are in the TimeSeries.
        Negative end times wrap around.
        """
        if start is None:
            start = self.start()
        if end is None:
            end = self.end()
        elif end < 0:
            end = self.end() + end
        assert start <= end, "start is larger than end!"
        return start, end
