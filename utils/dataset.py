import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
from utils.language_utils import word_to_indices, letter_to_vec


def batch_data(data, batch_size, seed):

    x = data['x']
    y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)

    # loop through mini-batches
    for i in range(0, len(x), batch_size):
        batched_x = x[i:i + batch_size]
        batched_y = y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


