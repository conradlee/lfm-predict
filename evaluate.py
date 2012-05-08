import numpy as np
import utils
from scipy import sparse

TRAIN_TEST_SPLIT_TIME = 1306713600
END_TIME = 1327881600

def train_test_vel_dicts(charts_type, norm="l1"):
    velocity_dict = utils.velocities_dict(charts_type, norm)
    train_velocities, test_velocities = {}, {}
    for (w1, w2), mat in velocity_dict.iteritems():
        if w2 < TRAIN_TEST_SPLIT_TIME:
            train_velocities[(w1, w1)] = mat
        elif w1 < END_TIME:
            test_velocities[(w1, w2)] = mat
    return train_velocities, test_velocities
