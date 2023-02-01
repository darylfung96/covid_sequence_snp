import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta


class KmersDataset:
    def __init__(self, k_mers_dict):
        self.k_mers_dict = k_mers_dict
        self.k_mers = self.k_mers_dict.keys()
        self.k_mers_counts = self.k_mers_dict.values()

    def get_k_mers(self):
        return self.k_mers

    def get_k_mers_counts(self):
        return self.k_mers_counts


@task(name='Read data', cache_key_fn=task_input_hash)
def read_data():
    data = pd.read_csv('0002.table', '\t')
    alt_alleles = data['ALT']
    return alt_alleles


@task(name='Preprocess data', cache_key_fn=task_input_hash)
def preprocess_data(alt_alleles):
    # alternate alleles sometimes have more than one base
    # we only select one base to be used in case if there are more than one
    alt_alleles = alt_alleles.map(lambda x: x.replace('*', '').replace(',', ''))
    alt_alleles = alt_alleles.map(lambda x: x[0])

    # create one hot
    one_hot_encoder = OneHotEncoder()
    alt_alleles = np.expand_dims(alt_alleles, 1)
    onehot_alleles = one_hot_encoder.fit_transform(alt_alleles)

    return onehot_alleles, one_hot_encoder


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

