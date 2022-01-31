from pandas import DataFrame
from math import inf

from .tdms_reader import Columns


def time_filter(data: DataFrame, start: float = -inf, end: float = inf):
    """Removes the data in the DataFrame that lies outsize of the given start and end time"""
    for i in range(len(data)):
        series_x = data.iloc[i][Columns.X]
        series_y = data.iloc[i][Columns.Y]
        mask = (start <= series_x) * (series_x <= end)
        data.iloc[i][Columns.X] = series_x[mask]
        data.iloc[i][Columns.Y] = series_y[mask]
