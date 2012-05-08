from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
import numpy as np
from scipy import io, sparse
import os
from glob import glob
import itertools

BINARY_PATH = "cached/binary/"
MM_PATH = "cached/mm/"
WEEK_LENGTH = 60 * 60 * 24 * 7 # (in seconds)

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
    diff = theta1 - theta2
    diff.data **=2
    return np.sqrt(diff.sum() / diff.shape[1])


### I/O Functions ###

def read_binary(chart_type, unix_timestamp, norm=None):
    fname = get_binary_filename(chart_type, unix_timestamp)
    if not(os.path.isfile(fname)):
        write_binary(chart_type, unix_timestamp)
    try:
        mat = io.loadmat(fname)["listen_matrix"].tocsr().astype("f8")
    # Hack to get around bug loading matlab matrices with no entries
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

def velocities_dict(chart_type, norm="l1"):
    # A valid week is one with at least 60% of the median activity level
    week_dict_raw = load_all(chart_type)
    week_dict_norm = load_all(chart_type, norm=norm)
    median_num_listens = np.median([mat.sum() for mat in week_dict_raw.values()])
    accepted_weeks = filter(lambda w: week_dict_raw[w].sum() > median_num_listens * 0.6, sorted(week_dict_raw.keys()))
    adjacent_accepted_weeks = [(w1, w2) for w1, w2 in pairwise(accepted_weeks) if (w2 - w1) == WEEK_LENGTH]
    return dict([((w1, w2), week_dict_norm[w2] - week_dict_norm[w1]) for w1, w2 in adjacent_accepted_weeks])

def split_current_past(velocities_dict, lag_timesteps):
    # generator that produces (current, (oldest, newer, ..., last_before_current))
    weeks = sorted(velocities_dict.keys(), key=lambda tup: tup[0])
    l, r = 0, lag_timesteps
    while r < len(weeks):
        expected_r = weeks[l][0] + WEEK_LENGTH * lag_timesteps
        if weeks[r][0] == expected_r:
            yield weeks[r], weeks[l:r]
        l += 1
        r += 1

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)
