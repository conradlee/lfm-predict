from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import numpy as np
from scipy import io, sparse
import os
from glob import glob
import itertools

BINARY_PATH = "cached/binary/"
MM_PATH = "cached/mm/"


### Measures ###

def rmse(theta1, theta2):
    """Root mean squared error regression loss

    Return a a positive floating point value (the best value is 0.0).

    Parameters
    ----------
    y_true : array-like
    
    y_pred : array-like

    Returns
    -------
    loss : float
    """
    return np.sqrt(mean_squared_error(theta1, theta2))


### I/O Functions ###

def read_binary(chart_type, unix_timestamp, norm=None):
    fname = get_binary_filename(chart_type, unix_timestamp)
    if not(os.path.isfile(fname)):
        write_binary(chart_type, unix_timestamp)
    try:
        mat = io.loadmat(fname)["listen_matrix"].tocsr().astype("f8")
    # Hack to get around bug loading matrices with no entries
    except ValueError:
        mm_fname = "%s%s/%d.mm" % (MM_PATH, chart_type, unix_timestamp)
        mat = io.mmread(mm_fname).tocsr().astype("d4")
        assert mat.nnz == 0
    if not(norm is None):
        mat = normalize(mat, norm, axis=1, copy=False)
    return mat
    
def write_binary(chart_type, unix_timestamp):
    mm_fname = "%s%s/%d.mm" % (MM_PATH, chart_type, unix_timestamp)
    listen_matrix = io.mmread(mm_fname).tocsr().astype("u4")
    binary_fname = get_binary_filename(chart_type, unix_timestamp)
    print "Writing binary to %s with %d nnz" % (binary_fname, listen_matrix.nnz)
    io.savemat(binary_fname, {"listen_matrix": listen_matrix})

def get_binary_filename(chart_type, unix_timestamp):
    p = BINARY_PATH + chart_type + "/"
    if not(os.path.isdir(p)):
        os.makedirs(p)
    return "%s%d.mat" % (p, unix_timestamp)

def load_all(chart_type, norm=None):
    fn_selector = "%s%s/*.mm" % (MM_PATH, chart_type)
    unix_timestamps = sorted([int(fn.split("/")[-1].split(".")[0]) for fn in glob(fn_selector)])
    return dict([(uts, read_binary(chart_type, uts, norm=norm)) for uts in unix_timestamps])


### Functions for selecting velocities from valid weeks ###

def get_intervals(chart_type, norm="l1"):

    valid_week_indicator = mark_valid_weeks(chart_type)

    # Group contiguous blocks of valid weeks
    intervals = []
    current_interval = []
    for week in sorted(valid_week_indicator.keys()):
        if valid_week_indicator[week]:
            current_interval.append(week)
        else:
            if len(current_interval) > 0:
                intervals.append(current_interval)
                current_interval = []
    if len(current_interval) > 0:
        intervals.append(current_interval)

    
    return intervals

def valid_velocities(chart_type, norm="l1"):
    intervals = get_intervals(chart_type, norm)
    normalized_matrices = load_all(chart_type, norm)
    for interval in intervals:
        for p1, p2 in pairwise(interval):
            vel_mat = normalized_matrices[p2] - normalized_matrices[p1]
            yield (p1, p2), vel_mat
    
def mark_valid_weeks(chart_type):
    """
    A week is considered valid if it has more global listens than
    half of the median number of global listens.
    """
    week_dict = load_all(chart_type)
    median_num_listens = np.median([mat.sum() for mat in  week_dict.values()])
    missing_dict = {}
    for unix_timestamp in sorted(week_dict.keys()):
        mat = week_dict[unix_timestamp]
        missing_dict[unix_timestamp] = mat.sum() > (median_num_listens * 0.60)
    return missing_dict

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)
