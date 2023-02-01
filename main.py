import pandas as pd
from prefect import flow, task
from prefect.tasks import task_input_hash
import torch
import torch.nn as nn

from data import read_data, preprocess_data, create_k_mers
from model import COVIDSeq1D, get_loss

@task(cache_key_fn=task_input_hash)
def augment_sequence(tensor_one_hot_alleles, counts):
    for i in range(counts):
        new = tensor_one_hot_alleles[:, :, torch.randperm(tensor_one_hot_alleles.shape[2])].clone()
        tensor_one_hot_alleles = torch.cat([tensor_one_hot_alleles, new], dim=0)
    return tensor_one_hot_alleles

@flow
def training_loop(model, inputs, labels):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = get_loss(outputs, labels)
        loss.backward()
        optimizer.step()


@flow(name='k_mers')
def k_mers_pipeline():
    alt_alleles = read_data()
    k_mers = create_k_mers(alt_alleles, 10)


@flow(name='normal')
def normal_pipeline():
    # retrieve data
    alt_alleles = read_data()

    # data preprocessing
    one_hot_alleles, one_hot_encoder = preprocess_data(alt_alleles)
    one_hot_alleles = one_hot_alleles.toarray()
    tensor_one_hot_alleles = torch.FloatTensor(one_hot_alleles).unsqueeze(0).permute(0, 2, 1)
    tensor_one_hot_alleles = augment_sequence(tensor_one_hot_alleles, 2)

    # model creation
    model = COVIDSeq1D(tensor_one_hot_alleles.shape[1], 4)
    training_loop(model, tensor_one_hot_alleles, torch.ones(tensor_one_hot_alleles.shape[0], 1))


k_mers_pipeline()