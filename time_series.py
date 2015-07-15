import math

import numpy as np
import pandas as pd

import scipy
import scipy.fftpack
from scipy.cluster.vq import kmeans, vq
from scipy.signal import argrelextrema

from rpy2.robjects import pandas2ri
pandas2ri.activate()

from smoothing import *
from r_packages import *


class TimeSeries:

    '''
    Class for analyzing of time series. Compute harmonic components, smoothed and reconstructed data, extremes.
    '''

    def __init__(self, data_list, dates, num_comp=3, num_harm=10, start_harm=2):
        '''
        Initialize instance of class with data list and corresponding date list.
        :param data_list: list of float
        :param dates: list of date
        :param num_comp: int, number of main periods (harmonic components)
        :param num_harm: int, number of first used harmonics
        :param start_harm: int, number of start harmonic
        :return:
        '''
        self.data_list = data_list
        self.dates = dates
        data_frame = pd.DataFrame(data={'Value': data_list}, index=dates)
        self.raw_data = data_frame['Value']
        self.num_comp = num_comp
        self.num_harm = num_harm
        self.start_harm = start_harm
        self.ssa = rssa.ssa(
            pd.DataFrame(data=list(self.raw_data), index=list(self.raw_data.index)))
        self._periods = {}  # will contain main periods
        # will contain dict of list of harmonics for every corresponding period
        self._period_harmonics = {}
        self._deviations = {}  # will contain standard deviation

    @property
    def deviations(self):
        '''
        Return std.deviations. Automatically call comp_main_periods() if needed.
        '''
        if not self._deviations:
            self.comp_main_periods()
        return self._deviations

    @property
    def periods(self):
        '''
        Return periods. Automatically call comp_main_periods() if needed.
        '''
        if not self._periods:
            self.comp_main_periods()
        return self._periods

    @property
    def period_harmonics(self):
        '''
        Return self._period_harmonics  Automatically call comp_main_periods() if needed.
        '''
        if not self._periods:
            self.comp_main_periods()
        return self._periods

    def get_raw_data(self):
        '''
        Return raw_data in pandas dataframe format.
        :return:
        '''
        return self.raw_data

    def _comp_periods(self):
        '''
        Return list of first n periods. Compute it by using Fourier transform to every reconstructed harmonic.
        :return: list of floats
        '''
        periods = []

        for i in range(self.start_harm, self.num_harm + 1):
            r = rssa.reconstruct(self.ssa, base.list(base.c(i)))
            signal = np.array(r[0][0])

            fft = abs(scipy.fft(signal))
            freqs = scipy.fftpack.fftfreq(signal.size)

            max_freq = abs(freqs[np.argmax(fft)])
            periods.append(1 / max_freq if max_freq != 0 else float('inf'))
        return periods

    def comp_main_periods(self):
        '''
        Return list of main periods. It's computed by clustering periods.
        :return: tuple of dicts:
                 dict with component number as key and main periods as value
                 dict with component number as key and numbers of harmonics as value
        '''
        periods = self._comp_periods()
        harm_data = np.vstack(np.array(periods))
        centroids, _ = kmeans(harm_data, self.num_comp)
        idx, _ = vq(harm_data, centroids)
        harm_numbers = []  # list of lists with number of harmonics
        # list of lists with periods for corresponding harmonics
        harm_periods = []
        for i in range(self.num_comp):
            harm_numbers.append([])
            harm_periods.append([])
            periods_copy = periods.copy()
            for item in harm_data[idx == i]:
                index = periods_copy.index(item)
                harm_numbers[i].append(index + self.start_harm)
                harm_periods[i].append(periods[index])
                periods_copy[index] = 0

        # sort main periods and components of period
        centroids_list = [item[0] for item in centroids]
        for i in range(self.num_comp):
            max_centroid_index = centroids_list.index(max(centroids_list))
            self.periods[i + 1] = centroids_list[max_centroid_index]
            self.period_harmonics[i + 1] = harm_numbers[max_centroid_index]
            self.deviations[
                i + 1] = np.std(np.array(harm_periods[max_centroid_index]))
            centroids_list[max_centroid_index] = 0
        return self.periods, self.period_harmonics, self.deviations

    def get_main_periods(self, order):
        '''
        Retrun main periods by order of components.
        :param order: tuple of int
        :return: dict with component number as keys and periods as values
        '''
        return {o: self.periods[o] for o in order}

    def get_deviations(self, order):
        '''
        Return standard deviations of main periods.
        :param order: tuple of int
        :return: dict with component number as keys and standard deviations as values
        '''
        return {o: self.deviations[o] for o in order}


    def get_smooth(self, comp_num):
        '''
        Compute smoothed graph by component.
        :param comp_num: int, component number
        :return: smoothed graph in pandas dataframe format
        '''
        if not self.periods:
            self.comp_main_periods()
        sm_window = self.periods[comp_num]
        smoothed = pd.rolling_mean(self.raw_data, sm_window, min_periods=0)
        return smoothed

    def get_reconstruct(self, comp_num):
        '''
        Compute reconstructed graph by component.
        :param comp_num: int, component number
        :return: reconstructed timeseries in pandas dataframe format
        '''
        comp_list = [
            1, ]  # include trend component to harmonic components for reconstruction
        for i in range(comp_num):
            comp_list = comp_list + self.period_harmonics[i + 1]
        r = rssa.reconstruct(self.ssa, base.list(base.c(*comp_list)))
        row = r[0][0]
        row_frame = pd.DataFrame(data={'Value': row}, index=self.dates)
        reconstructed = row_frame['Value']
        return reconstructed

    def get_recon_harm(self, comp_num):
        '''
        Compute reconstructed harmonics by component.
        :param comp_num: int, component number
        :return: reconsturcted timeseries in pandas dataframe format
        '''
        comp_list = self.period_harmonics[comp_num]
        r = rssa.reconstruct(self.ssa, base.list(base.c(*comp_list)))
        row = r[0][0]
        row_frame = pd.DataFrame(data={'Value': row}, index=self.dates)
        reconstructed = row_frame['Value']
        return reconstructed

    def local_extremes(self, timeseries, golay_window=51, golay_order=3):
        '''
        Defines local maximum and minimums of timeseries.
        Return maximums and minimums in pandas dataframe format.
        :param timeseries: timeseries in pandas dataframe format, timeseries for looking of extremes
        :param golay_window: int, window of smoothing for savitzky_golay filter
        :param golay order: int, polynomial order of savitzky_golay filter
        :return: tuple of dataframe
        '''
        if golay_window % 2 == 0:
            golay_window = golay_window + 1
        smooths = savitzky_golay(timeseries, golay_window, golay_order)
        # compute maximums
        maxs = argrelextrema(smooths, np.greater)
        max_series = timeseries[maxs[0]]
        # compute minimums
        mins = argrelextrema(smooths, np.less)
        min_series = timeseries[mins[0]]
        return max_series, min_series

    def trend(self, last_max, last_min, current_point):
        '''
        Compute and return trend rate and trend angle from last extreme.
        :param last_max: tuple of date and float
        :param last_min: tuple of date and float
        :param current_point: tuple of date and float
        :return: tuple of str, float and float, type of extreme, trend rate, trend angle
        '''
        if last_max[0] > last_min[0]:
            extreme_type = 'max'
            last_extreme = last_max
        else:
            extreme_type = 'min'
            last_extreme = last_min

        days_delta = len(self.raw_data[self.raw_data.index > last_extreme[0]])
        trend_rate = (current_point[1] - last_extreme[1]) / days_delta
        trend_angle = 10000 * 180 * (math.atan(trend_rate)) / math.pi
        return extreme_type, trend_rate, trend_angle
