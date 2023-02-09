import pandas as pd
from argparse import ArgumentParser
from prefect import flow, task
from prefect.tasks import task_input_hash
import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import shap
from glob import glob

from data import read_data, preprocess_data, create_k_mers, SeqDataset
from model import COVIDSeq1D, COVIDSeq1DLSTM, get_loss


def set_seeds(seed):
    # set seeds
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@task(cache_key_fn=task_input_hash)
def augment_sequence(tensor_one_hot_alleles, seed):
    set_seeds(seed)
    return tensor_one_hot_alleles[:, :, torch.randperm(tensor_one_hot_alleles.shape[2])].clone()


def generate_label(seed):
    set_seeds(seed)
    return torch.randint(0, 2, size=[1, 1]).float()



@flow
def training_loop(model, inputs):
    inputs.set_train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_augments = 10

    for i in range(1):
        # for j in range(num_augments):
        for item, label in inputs:
            optimizer.zero_grad()
            # augmented_inputs = augment_sequence(item, j)
            # label = generate_label(j)
            outputs = model(item)
            loss = get_loss(outputs, label)
            print(loss)
            loss.backward()
            optimizer.step()

@flow
def validation_loop(model, inputs):
    inputs.set_train(False)
    all_outputs = []
    all_labels = []
    for item, label in inputs:
        # augmented_inputs = augment_sequence(item, j)
        # label = generate_label(j)
        outputs = model(item)
        all_outputs.append(outputs.detach().numpy())
        all_labels.append(label.detach().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    fpr, tpr, threshold = roc_curve(all_labels.ravel(), all_outputs.ravel())
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]

    all_outputs[all_outputs > optimal_threshold] = 1
    all_outputs[all_outputs <= optimal_threshold] = 0

    fpr, tpr, threshold = roc_curve(all_labels.ravel(), all_outputs.ravel())
    roc_auc = auc(fpr, tpr)
    print(f'ROC value: {roc_auc}')

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('validation_roc_plot.png')
    plt.show()


@flow
def interpretable_shap(model, inputs):
    # unsqueeze to make 3 dimension since shap only allows 2 dimensions at max
    shap_model = lambda x: model(Variable(torch.from_numpy(x)).unsqueeze(1)).detach().numpy()
    explainer = shap.KernelExplainer(shap_model, inputs.detach().numpy())
    shap_values = explainer.shap_values(inputs.detach().numpy())
    shap.summary_plot(shap_values[0])
    return shap_values


@flow(name='k_mers')
def k_mers_pipeline():
    alt_alleles = read_data()
    k_mers = create_k_mers(alt_alleles, 10)
    print()


@flow(name='normal')
def normal_pipeline():

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--table_folder', type=str)
    args = arg_parser.parse_args()

    all_samples = glob(args.table_folder)
    seq_dataset = SeqDataset(all_samples)

    # model creation
    model = COVIDSeq1D(seq_dataset[0][0].shape[1])
    training_loop(model, seq_dataset)
    validation_loop(model, seq_dataset)

    # convert tensor to shap dimension
    # if tensor_processed_alleles.shape[1] == 1:
    #     shap_processed_alleles = tensor_processed_alleles.squeeze(1)
    # else:
    #     shap_processed_alleles = tensor_processed_alleles
    #
    # shap_values = interpretable_shap(model, shap_processed_alleles)
    # print()


normal_pipeline()

#TODO: generate ROC plot to show performance
#TODO: show some visualization plot (to show on this chromosome any specific region that shows high predictive ability)
#TODO: 1d CNN + LSTM (show ROC and visualization)
#TODO: try OneHot vs Label
