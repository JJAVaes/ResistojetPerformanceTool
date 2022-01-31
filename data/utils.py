from typing import Optional

from pandas import Series, DataFrame

from .tdms_reader import Columns
from analysis.time_series import TimeSeries


def get_channel_time_and_values(data: DataFrame, channel: str):
    """Returns the timestamps and values of the given channel"""
    time = get_channel_value(data, channel, Columns.X)
    values = get_channel_value(data, channel, Columns.Y)
    return time, values


def get_channel_value(data: DataFrame, channel: str, column: str):
    """Gets the data from the DataFrame given the channel title and column"""
    return get_value(get_channel(data, channel), column)


def get_value(data: DataFrame, columnm: str):
    """Gets the data from the DataFrame given the column"""
    return data[columnm].values[0]


def get_channel(data: DataFrame, channel: str) -> DataFrame:
    """Return a DataFrame containing only the given channel"""
    return data.loc[data[Columns.TITLE] == channel]


def unpack_series(series: Series) -> Series:
    """
    Unpacks a pandas.Series that looks like:
        n   [a, b, c, d, e, f]
    To:
        0   a
        1   b
        ...
        5   f
    """
    return Series(series.values[0])


def get_time_series(data: DataFrame, channel: str, name: str = None) -> Optional[TimeSeries]:
    """Get the data from the given channel and return in a TimeSeries object"""
    channel_data = get_channel(data, channel)

    if channel_data.empty:
        # Channel does not exist in the DataFrame
        return None

    if not name:
        name = channel_data[Columns.TITLE].values[0]
    units = channel_data[Columns.Y_UNIT].values[0]

    time, values = get_channel_time_and_values(data, channel)
    return TimeSeries(time=time, values=values, name=name, unit=units)
