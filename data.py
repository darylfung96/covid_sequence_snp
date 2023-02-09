import numpy as np
import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from prefect import task, flow
from prefect.tasks import task_input_hash
from datetime import timedelta
from torch.utils.data import Dataset


class KmersDataset:
    def __init__(self, k_mers_dict):
        self.k_mers_dict = k_mers_dict
        self.k_mers = self.k_mers_dict.keys()
        self.k_mers_counts = self.k_mers_dict.values()

    def get_k_mers(self):
        return self.k_mers

    def get_k_mers_counts(self):
        return self.k_mers_counts


class SeqDataset(Dataset):
    def __init__(self, filenames, encoding_type='discrete', test_split=0.2, train=True):
        self.filenames = filenames
        self.encoding_type = encoding_type
        self.encoder = None

        # shuffle and get train test splits
        test_size = int(len(self.filenames) * test_split)
        train_size = len(self.filenames) - test_size

        self.train_filenames = self.filenames[:train_size]
        self.test_filenames = self.filenames[train_size:]

        self.labels_dict = self._read_labels()

        self.train = train

    def _read_labels(self):
        labels_dict = {}
        with open('positive_sample_id.txt', 'r') as file:
            positive_labels = file.read().split('\n')
            for positive_label in positive_labels:
                labels_dict[positive_label] = [1]
        with open('negative_sample_id.txt', 'r') as file:
            negative_labels = file.read().split('\n')
            for negative_label in negative_labels:
                labels_dict[negative_label] = [0]
        return labels_dict

    def set_train(self, value):
        self.train = value

    def __len__(self):
        return len(self.train_filenames) if self.train else len(self.test_filenames)

    def __getitem__(self, idx):
        sample_list = self.train_filenames if self.train else self.test_filenames

        alt_alleles = read_data(sample_list[idx])
        label = self.labels_dict[os.path.basename(sample_list[idx])]
        processed_alleles, encoder = preprocess_data(alt_alleles, self.encoding_type, self.encoder)
        if self.encoder is None:
            self.encoder = encoder
        return torch.FloatTensor(processed_alleles), torch.FloatTensor(label).unsqueeze(0)


@task(name='Read data', cache_key_fn=task_input_hash)
def read_data(filename):
    data = pd.read_csv(filename, '\t')
    alt_alleles = data['ALT']
    return alt_alleles


@task(cache_key_fn=task_input_hash)
def create_one_hot(alt_alleles, one_hot_encoder=None):
    alt_alleles = np.expand_dims(alt_alleles, 1)

    if one_hot_encoder is None:
        # create one hot
        one_hot_encoder = OneHotEncoder()
        onehot_alleles = one_hot_encoder.fit_transform(alt_alleles)
    else:
        onehot_alleles = one_hot_encoder.transform(alt_alleles)
    tensor_onehot_alleles = torch.FloatTensor(onehot_alleles.toarray()).unsqueeze(0).permute(0, 2, 1)
    return tensor_onehot_alleles, one_hot_encoder


@task(cache_key_fn=task_input_hash)
def create_discrete(alt_alleles, label_encoder=None):
    if len(alt_alleles.shape) != 2:
        alt_alleles = np.expand_dims(alt_alleles, 1)

    if label_encoder is None:
        # create one hot
        label_encoder = LabelEncoder()
        discrete_alleles = label_encoder.fit_transform(alt_alleles.ravel())
    else:
        discrete_alleles = label_encoder.transform(alt_alleles.ravel())

    if len(discrete_alleles.shape) < 2:
        discrete_alleles = np.expand_dims(discrete_alleles, 0)
        discrete_alleles = np.expand_dims(discrete_alleles, 0)

    return torch.FloatTensor(discrete_alleles), label_encoder


def preprocess_data(alt_alleles, encoding_type='discrete', encoder=None):
    # alternate alleles sometimes have more than one base
    # we only select one base to be used in case if there are more than one
    alt_alleles = alt_alleles.map(lambda x: x.replace('*', '').replace(',', ''))
    alt_alleles = alt_alleles.map(lambda x: x[0])

    preprocess_models = {
        'onehot': create_one_hot,
        'discrete': create_discrete
    }

    processed_alleles, encoder = preprocess_models[encoding_type](alt_alleles, encoder)

    return processed_alleles, encoder


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(minutes=1))
def create_k_mers(alleles, count):
    k_mers_dict = {}
    for i in range(0, alleles.shape[0], count):
        current_alleles = ''.join(alleles[i:i+count].values)
        current_alleles = current_alleles.replace('*', '')
        current_alleles = current_alleles.replace(',', '')

        if k_mers_dict.get(current_alleles, None) is None:
            k_mers_dict[current_alleles] = 1
        else:
            k_mers_dict[current_alleles] += 1
    return k_mers_dict

