import numpy as np
import xarray as xr
from constants import *

def remove_nan_inf(data, labels, echogram, frequencies, new_value=0.0):
    '''
    Reassigns all non-finite data values (nan, positive inf, negative inf) to new_value.
    :param data:
    :param labels:
    :param echogram:
    :param new_value:
    :return:
    '''
    labels[np.invert(np.isfinite(data[0, :, :]))] = LABEL_IGNORE_VAL
    data[np.invert(np.isfinite(data))] = new_value
    return data, labels, echogram, frequencies

def remove_nan_inf_xr(data, labels, echogram, frequencies, new_value=0.0):
    data = xr.where(data.isnull(), new_value, data)
    return data, labels, echogram, frequencies

def set_data_border_value(data, labels, echogram, frequencies, border_value = 0.0):
    """ Set data points outside data border to border_value """
    data[:, labels == LABEL_BOUNDARY_VAL] = border_value
    return data, labels, echogram, frequencies

def db_with_limits(data, labels, echogram, frequencies, limit_low=-75, limit_high=0):
    data = db(data)
    data[data > limit_high] = limit_high
    data[data < limit_low] = limit_low
    return data, labels, echogram, frequencies

def db_with_limits_scaled(data, labels, echogram, frequencies, limit_low=-75, limit_high=0):
    data = db(data)
    data[data > 0] = 0
    data[data > limit_high] = limit_high
    data[data < limit_low] = limit_low
    data = 1 + data / np.abs(limit_low)
    return data, labels, echogram, frequencies

def db(data, eps=1e-10):
    """ Decibel (log) transform """
    return 10 * np.log10(data + eps)