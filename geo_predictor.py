import os
import numpy as np
import utils
import evaluate


def prepare_data(velocities_dict, selected_city, included_city_idxs, n_lag_timesteps, col_labels=None):
    current_past_timedeltas = list(utils.split_current_past(velocities_dict, n_lag_timesteps))
    n_examples = 0

    # Determine number of rows in traning data
    for current_td, past_tds in current_past_timedeltas:
        selected_artists = velocities_dict[current_td][selected_city,:].nonzero()[1]
        n_examples += len(selected_artists)
            
    # Map (city_idx, lag_size) tups to column number
    if col_labels:
        col_idx_map = dict([(t, i) for i, t in enumerate(col_labels)])
    else:
        col_idx_map = {}
        for city in included_city_idxs:
            for weeks_offset in range(n_lag_timesteps):
                col_idx_map[(city, weeks_offset + 1)] = len(col_idx_map)

    # Initialize and populate train matrices
    train_X = np.zeros((n_examples, len(col_idx_map)), dtype="f8")
    train_Y = np.zeros(n_examples, dtype="f8")
    row_labels = []
    start_row = 0    

    for current_td, previous_tds in current_past_timedeltas:
        current_velocity = velocities_dict[current_td]
        predicted_artists = current_velocity[selected_city,:].nonzero()[1]
        end_row = start_row + len(predicted_artists)
        for previous_td in previous_tds:
            lagged_vel_mat = velocities_dict[previous_td].tocsc()[:,predicted_artists].todense()
            weeks_offset = (current_td[0] - previous_td[0]) / utils.WEEK_LENGTH
            for city_idx in included_city_idxs:
                col_idx = col_idx_map[(city_idx, weeks_offset)]
                train_X[start_row:end_row, col_idx] = lagged_vel_mat[city_idx,:].ravel()
        train_Y[start_row:end_row] = current_velocity.tocsc()[selected_city,predicted_artists].todense().ravel()
        start_row += len(predicted_artists)

        for a in predicted_artists:
            row_labels.append((current_td, a))

    # Maps column index to (city, offset) pair
    idx2col = dict([[v, k] for k, v in col_idx_map.items()])
    col_labels = [idx2col[i] for i in range(len(col_idx_map))]
    
    return train_X, train_Y, col_labels, row_labels  

