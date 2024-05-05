# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:05:24 2021

@author: Ranak Roy Chowdhury
@modified: Erik Novak
"""
import math
import random
import warnings

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import src.models.multitask_transformer_class as multitask_transformer_class

warnings.filterwarnings("ignore")

# -----------------------------------------------
# Hyperparameter preparation
# -----------------------------------------------


# prepare the hyperparameters
def prepare_hyperparameters(args):
    return {
        "dataset": args.dataset,
        "model": args.model,
        "batch": args.batch,
        "lr": args.lr,
        "sparsity": args.sparsity,
        "epochs": args.epochs,
        "nlayers": args.nlayers,
        "emb_size": args.emb_size,
        "nhead": args.nhead,
        "task_rate": args.task_rate,
        "masking_ratio": args.masking_ratio,
        "lamb": args.lamb,
        "ratio_highest_attention": args.ratio_highest_attention,
        "dropout": args.dropout,
        "avg": args.avg,
        "nhid": args.nhid,
        "nhid_task": args.nhid_task,
        "nhid_tar": args.nhid_tar,
    }


def get_prop(args):
    return prepare_hyperparameters(args)


# -----------------------------------------------
# Data preparation
# -----------------------------------------------


# THE SPARSIFY FUNCTIONS
# ----------------------
# The first function is used to sparsify the time series values at random
# The second function is used to sparsify the time series values using burst sparsification (in batches)


def random_sparsify_time_series(time_series, p):
    """Random sparsification of time series"""
    number_of_missing = int(p * len(time_series))
    idx = list(range(len(time_series)))
    random.shuffle(idx)
    for j in range(number_of_missing):
        time_series[idx[j]] = 0
    return time_series


def burst_sparsify_time_series(time_series, p):
    """Burst sparsification of time series"""
    number_of_missing = int(p * len(time_series))
    missing_from = random.randint(0, len(time_series) - number_of_missing)
    time_series[missing_from : missing_from + number_of_missing] = [
        0
    ] * number_of_missing
    return time_series


def sparsify_time_series_dataset(ts_data, p, seed=1):
    random.seed(seed)
    for i in range(len(ts_data)):
        #! Change the function to change the sparsification method
        ts_data[i] = burst_sparsify_time_series(ts_data[i], p)
    return ts_data


def format_labels(labels):
    min_label = min(labels)
    max_label = max(labels)

    if min_label == 0:
        n_classes = int(max_label + 1)
    elif min_label == 1:
        labels = labels - min_label
        n_classes = int(max_label)
    elif min_label == -1:
        if np.sum(labels == -1) + np.sum(labels == 1) == len(labels):
            n_classes = 2
            labels[labels == -1] = 0
        else:
            raise Exception("Unexpected labels!")
    else:
        raise Exception("Unexpected labels!")

    return np.array(labels).astype(int), n_classes


def data_loader(data_path, sparsity):
    train_data_with_labels = np.genfromtxt(f"{data_path}_TRAIN.txt")
    test_data_with_labels = np.genfromtxt(f"{data_path}_TEST.txt")

    data_with_labels = np.vstack((train_data_with_labels, test_data_with_labels))

    data = data_with_labels[:, 1:]
    data = sparsify_time_series_dataset(data, sparsity)

    labels = data_with_labels[:, 0]
    labels, n_classes = format_labels(labels)

    return data, labels, n_classes


def make_perfect_batch(X, batch_size):
    extension = torch.zeros((batch_size - X.shape[0], X.shape[1]))
    X = torch.cat((X, extension), dim=0)
    return X


def mean_standardize_fit(X):
    m1 = np.mean(X, axis=1)
    mean = np.mean(m1, axis=0)

    s1 = np.std(X, axis=1)
    std = np.mean(s1, axis=0)

    return mean, std


def mean_standardize_transform(X, mean, std):
    return (X - mean) / std


def preprocess(X_train, y_train, X_test, y_test):
    X_train = np.array(X_train).astype(float)
    X_test = np.array(X_test).astype(float)

    mean, std = mean_standardize_fit(X_train)
    X_train = mean_standardize_transform(X_train, mean, std)
    X_test = mean_standardize_transform(X_test, mean, std)

    X_train = torch.Tensor(X_train).float()
    X_test = torch.Tensor(X_test).float()

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_test, y_test


# -----------------------------------------------
# Model preparation
# -----------------------------------------------


def initialize_model_objects(prop):
    model = multitask_transformer_class.MultitaskTransformerModel(
        prop["device"],
        prop["n_classes"],
        prop["seq_len"],
        prop["batch"],
        prop["input_size"],
        prop["emb_size"],
        prop["nhead"],
        prop["nhid"],
        prop["nhid_tar"],
        prop["nhid_task"],
        prop["nlayers"],
        prop["dropout"],
    ).to(prop["device"])

    criterion_tar = torch.nn.MSELoss()
    criterion_task = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=prop["lr"])

    return model, optimizer, criterion_tar, criterion_task


def attention_sampled_masking_heuristic(
    X, masking_ratio, ratio_highest_attention, instance_weights
):
    # instance_weights = torch.sum(attention_weights, axis = 1)
    res, index = instance_weights.topk(
        int(math.ceil(ratio_highest_attention * X.shape[1]))
    )
    index = index.cpu().data.tolist()
    index2 = [
        random.sample(index[i], int(math.ceil(masking_ratio * X.shape[1])))
        for i in range(X.shape[0])
    ]
    return np.array(index2)


def random_instance_masking(
    X, masking_ratio, ratio_highest_attention, instance_weights
):
    indices = attention_sampled_masking_heuristic(
        X, masking_ratio, ratio_highest_attention, instance_weights
    )
    boolean_indices_masked = np.array(
        [
            [True if i in index else False for i in range(X.shape[1])]
            for index in indices
        ]
    )

    boolean_indices_unmasked = np.invert(boolean_indices_masked)
    X_train_tar = torch.Tensor(np.where(boolean_indices_unmasked, X, 0.0))
    y_train_tar_masked = X[boolean_indices_masked].reshape(X.shape[0], -1)
    y_train_tar_unmasked = X[boolean_indices_unmasked].reshape(X.shape[0], -1)

    return (
        X_train_tar,
        y_train_tar_masked,
        y_train_tar_unmasked,
        boolean_indices_masked,
        boolean_indices_unmasked,
    )


def compute_tar_loss(
    model,
    device,
    criterion_tar,
    y_train_tar_masked,
    y_train_tar_unmasked,
    batched_input_tar,
    batched_boolean_indices_masked,
    batched_boolean_indices_unmasked,
):
    out_tar = model(batched_input_tar.unsqueeze(2).to(device), "reconstruction")[0]
    out_tar_masked = (
        out_tar[batched_boolean_indices_masked]
        .reshape(y_train_tar_masked.shape[0], -1)
        .cpu()
    )
    out_tar_unmasked = (
        out_tar[batched_boolean_indices_unmasked]
        .reshape(y_train_tar_unmasked.shape[0], -1)
        .cpu()
    )

    loss_tar_masked = criterion_tar(out_tar_masked, y_train_tar_masked)
    loss_tar_unmasked = criterion_tar(out_tar_unmasked, y_train_tar_unmasked)

    return loss_tar_masked, loss_tar_unmasked


def compute_task_loss(
    n_classes,
    model,
    device,
    criterion_task,
    y_train_task,
    batched_input_task,
):
    out_task, attn = model(batched_input_task.unsqueeze(2).to(device), "classification")
    out_task = out_task[: y_train_task.shape[0], :].view(-1, n_classes).cpu()
    loss_task = criterion_task(out_task, y_train_task)  # dtype = torch.long
    return attn.cpu(), loss_task


def multitask_train(
    model,
    criterion_tar,
    criterion_task,
    optimizer,
    X_train_tar,
    X_train_task,
    y_train_tar_masked,
    y_train_tar_unmasked,
    y_train_task,
    boolean_indices_masked,
    boolean_indices_unmasked,
    prop,
):
    model.train()  # Turn on the train mode
    total_loss_tar_masked, total_loss_tar_unmasked, total_loss_task = 0.0, 0.0, 0.0
    attn_arr = []

    train_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_train_task),
        torch.LongTensor(y_train_task),
        torch.Tensor(X_train_tar),
        torch.Tensor(y_train_tar_masked),
        torch.Tensor(y_train_tar_unmasked),
        torch.BoolTensor(boolean_indices_masked),
        torch.BoolTensor(boolean_indices_unmasked),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=prop["batch"], shuffle=True
    )

    for data in train_loader:
        optimizer.zero_grad()

        X_train_task_batch = make_perfect_batch(data[0], prop["batch"])
        y_train_task_batch = data[1]
        X_train_tar_batch = make_perfect_batch(data[2], prop["batch"])
        y_train_tar_masked_batch = data[3]
        y_train_tar_unmasked_batch = data[4]
        boolean_indices_masked_batch = (
            make_perfect_batch(data[5], prop["batch"])
            .unsqueeze(2)
            .type(dtype=torch.bool)
        )
        boolean_indices_unmasked_batch = (
            make_perfect_batch(data[6], prop["batch"])
            .unsqueeze(2)
            .type(dtype=torch.bool)
        )

        loss_tar_masked, loss_tar_unmasked = compute_tar_loss(
            model,
            prop["device"],
            criterion_tar,
            y_train_tar_masked_batch,
            y_train_tar_unmasked_batch,
            X_train_tar_batch,
            boolean_indices_masked_batch,
            boolean_indices_unmasked_batch,
        )

        attn, loss_task = compute_task_loss(
            prop["n_classes"],
            model,
            prop["device"],
            criterion_task,
            y_train_task_batch,
            X_train_task_batch,
        )
        attn = attn.detach().cpu()
        total_loss_tar_masked += loss_tar_masked.item()
        total_loss_tar_unmasked += loss_tar_unmasked.item()
        total_loss_task += loss_task.item() * y_train_task_batch.shape[0]

        loss = (
            prop["task_rate"]
            * (prop["lamb"] * loss_tar_masked + (1 - prop["lamb"]) * loss_tar_unmasked)
            + (1 - prop["task_rate"]) * loss_task
        )
        loss.backward()
        optimizer.step()

        # remove the diagonal values of the attention map while aggregating the column wise attention scores
        attn_arr.append(
            torch.sum(attn, axis=1) - torch.diagonal(attn, offset=0, dim1=1, dim2=2)
        )

    instance_weights = torch.cat(attn_arr, axis=0)
    return (
        total_loss_tar_masked,
        total_loss_tar_unmasked,
        total_loss_task / y_train_task.shape[0],
        instance_weights,
    )


# -----------------------------------------------
# Training method
# -----------------------------------------------


def training(
    model,
    optimizer,
    criterion_tar,
    criterion_task,
    X_train_task,
    y_train_task,
    prop,
):
    instance_weights = torch.rand(X_train_task.shape[0], prop["seq_len"])

    for epoch in range(1, prop["epochs"] + 1):
        (
            X_train_tar,
            y_train_tar_masked,
            y_train_tar_unmasked,
            boolean_indices_masked,
            boolean_indices_unmasked,
        ) = random_instance_masking(
            X_train_task,
            prop["masking_ratio"],
            prop["ratio_highest_attention"],
            instance_weights,
        )

        (
            tar_loss_masked,
            tar_loss_unmasked,
            task_loss,
            instance_weights,
        ) = multitask_train(
            model,
            criterion_tar,
            criterion_task,
            optimizer,
            X_train_tar,
            X_train_task,
            y_train_tar_masked,
            y_train_tar_unmasked,
            y_train_task,
            boolean_indices_masked,
            boolean_indices_unmasked,
            prop,
        )

        tar_loss = tar_loss_masked + tar_loss_unmasked
        print(
            f"Epoch: {epoch:<3.0f} - TAR Loss: {tar_loss:<20.16f} - TASK Loss: {task_loss:<20.16f}"
        )


# -----------------------------------------------
# Evaluation method
# -----------------------------------------------


# might not be required if using a logger method
def store_prediction(instance, prediction, target, prop):
    with open(
        f"results/predictions_lr={str(prop['lr'])}.csv", mode="a", encoding="utf-8"
    ) as f:
        f.write(
            f"{prop['dataset']},{prop['model']},{prop['kfold']},{instance},{prediction},{target}\n"
        )


# might not be required if using a logger method
def store_accuracy(accuracy, prop):
    with open(
        f"results/accuracy_lr={str(prop['lr'])}.csv", mode="a", encoding="utf-8"
    ) as f:
        f.write(f"{prop['dataset']},{prop['model']},{prop['kfold']},{accuracy}\n")


def evaluate(y_pred, y, prop):

    pred, target = y_pred.cpu().data.numpy(), y.cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    acc = accuracy_score(target, pred)
    prec = precision_score(target, pred, average=prop["avg"])
    rec = recall_score(target, pred, average=prop["avg"])
    f1 = f1_score(target, pred, average=prop["avg"])

    instance = 0
    for p, t in zip(pred, target):
        store_prediction(instance, p, t, prop)
        instance += 1
    store_accuracy(acc, prop)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def test(model, test_data, test_targets, prop):
    model.eval()  # Turn on the evaluation mode

    train_dataset = torch.utils.data.TensorDataset(
        torch.Tensor(test_data),
        torch.LongTensor(test_targets),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=prop["batch"], shuffle=True
    )

    output_arr = []
    target_arr = []
    with torch.no_grad():
        for data in train_loader:

            X_batch = make_perfect_batch(data[0], prop["batch"])
            y_batch = data[1]

            out = model(X_batch.unsqueeze(2).to(prop["device"]), "classification")[0]
            out = out[: y_batch.shape[0], :].cpu()
            output_arr.append(out)
            target_arr.append(y_batch)

    return evaluate(torch.cat(output_arr, 0), torch.cat(target_arr, 0), prop)
