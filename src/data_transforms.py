import numpy as np
import xarray as xr
from utils import CombineFunctions
from constants import *


def get_data_transform_function(data_transforms):
    transform_functions = []
    for transform in data_transforms:
        transform_function = get_transform_function(transform)
        transform_functions.append(transform_function)
    
    return CombineFunctions(transform_functions)

def get_transform_function(transform):
    if transform == 'remove_nan_inf':
        return remove_nan_inf
    elif transform == 'remove_nan_inf_xr':
        return remove_nan_inf_xr
    elif transform == 'set_data_border_value':
        return set_data_border_value
    elif transform == 'db_with_limits':
        return db_with_limits
    elif transform == 'db_with_limits_scaled':
        return db_with_limits_scaled
    elif transform == 'db':
        return db
    else:
        raise ValueError(f"Unknown data transformation: {transform}")


def remove_nan_inf(data, labels, frequencies, new_value=0.0):
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
    return data, labels, frequencies

def remove_nan_inf_xr(data, labels, frequencies, new_value=0.0):
    data = xr.where(data.isnull(), new_value, data)
    return data, labels, frequencies

def set_data_border_value(data, labels, frequencies, border_value = 0.0):
    """ Set data points outside data border to border_value """
    data[:, labels == LABEL_BOUNDARY_VAL] = border_value
    return data, labels, frequencies

def db_with_limits(data, labels, frequencies, limit_low=-75, limit_high=0):
    data = db(data)
    data[data > limit_high] = limit_high
    data[data < limit_low] = limit_low
    return data, labels, frequencies

def db_with_limits_scaled(data, labels, frequencies, limit_low=-75, limit_high=0):
    data = db(data)
    data[data > 0] = 0
    data[data > limit_high] = limit_high
    data[data < limit_low] = limit_low
    data = 1 + data / np.abs(limit_low)
    return data, labels, frequencies

def db(data, eps=1e-10):
    """ Decibel (log) transform """
    return 10 * np.log10(data + eps)