import pandas as pd
from argparse import ArgumentParser
from prefect import flow, task
from prefect.tasks import task_input_hash
import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shap
import wandb
from glob import glob

from data import read_data, preprocess_data, create_k_mers, SeqDataset
from model import COVIDSeq1D, COVIDSeq1DLSTM, get_loss


device = 'cpu' if torch.cuda.is_available() else 'cpu'


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
def training_loop(model, inputs, batch_size):
    inputs.set_train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # num_augments = 10

    loader = DataLoader(inputs, batch_size=batch_size)
    for i in range(10):
        # for j in range(num_augments):
        for item, label in loader:
            optimizer.zero_grad()
            # augmented_inputs = augment_sequence(item, j)
            # label = generate_label(j)
            item = item.to(device)
            label = label.to(device)
            outputs = model(item)
            loss = get_loss(outputs, label)
            loss.backward()
            optimizer.step()
            wandb.log({'training_loss': loss.item()})

        # get validation loss
        inputs.set_train(False)
        all_val_loss = []
        loader = DataLoader(inputs, batch_size=1)
        for item, label in loader:
            # augmented_inputs = augment_sequence(item, j)
            # label = generate_label(j)
            item = item.to(device)
            label = label.to(device)
            outputs = model(item)
            loss = get_loss(outputs, label)
            all_val_loss.append(loss.item())
        avg_val_loss = sum(all_val_loss)/len(all_val_loss)
        wandb.log({'validation_loss': avg_val_loss})

@flow
def validation_loop(model, inputs, args):
    inputs.set_train(False)
    all_outputs = []
    all_labels = []

    loader = DataLoader(inputs, batch_size=1)
    for item, label in loader:
        # augmented_inputs = augment_sequence(item, j)
        # label = generate_label(j)
        item = item.to(device)
        label = label.to(device)
        outputs = model(item)
        all_outputs.append(outputs.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    os.makedirs('outputs', exist_ok=True)
    torch.save({
        'outputs': all_outputs,
        'labels': all_labels
    }, os.path.join('outputs', f'{args.model_type}_{args.encoding_type}'))

    fpr, tpr, threshold = roc_curve(all_labels.ravel(), all_outputs.ravel())
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    if optimal_threshold > 1:
        optimal_threshold -= 1

    all_outputs[all_outputs > optimal_threshold] = 1
    all_outputs[all_outputs <= optimal_threshold] = 0

    fpr, tpr, threshold = roc_curve(all_labels.ravel(), all_outputs.ravel())
    roc_auc = auc(fpr, tpr)
    wandb.log({'roc_auc': roc_auc, 'optimal_threshold': optimal_threshold})
    print(f'ROC value: {roc_auc}')

    f1 = f1_score(all_labels.ravel(), all_outputs.ravel())
    precision = precision_score(all_labels.ravel(), all_outputs.ravel())
    recall = recall_score(all_labels.ravel(), all_outputs.ravel())

    wandb.log({'roc_auc': roc_auc, 'optimal_threshold': optimal_threshold,
               'f1': f1, 'precision': precision, 'recall': recall})

    print(f'f1: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'validation_roc_plot_{inputs.encoding_type}.png')
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
    arg_parser.add_argument('--encoding_type', type=str, choices=['discrete', 'onehot'])
    arg_parser.add_argument('--model_type', type=str, default='conv1d_lstm', choices=['conv1d', 'conv1d_lstm'])
    arg_parser.add_argument('--batch_size', type=int, default=2)
    args = arg_parser.parse_args()
    wandb.init(group=args.model_type, name=args.encoding_type)

    all_samples = glob(args.table_folder)
    seq_dataset = SeqDataset(all_samples, encoding_type=args.encoding_type)

    # model creation
    model_dict = {
        'conv1d': COVIDSeq1D,
        'conv1d_lstm': COVIDSeq1DLSTM
    }

    random.seed(100)
    torch.random.manual_seed(100)
    torch.use_deterministic_algorithms(True)
    np.random.seed(100)

    model = model_dict[args.model_type](seq_dataset[0][0].shape[0])
    wandb.watch(model)
    model = model.to(device)
    training_loop(model, seq_dataset, args.batch_size)
    validation_loop(model, seq_dataset, args)

    torch.save(model.state_dict(), f"models/{args.encoding_type}.pt")
    art = wandb.Artifact(args.model_type, type="model")
    art.add_file(f"models/{args.encoding_type}.pt")
    wandb.log_artifact(art)

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
