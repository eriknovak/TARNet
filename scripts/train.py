# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
@modified: Erik Novak
"""
import argparse
import warnings
import pathlib
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

import src.utils.utils as utils

warnings.filterwarnings("ignore")

PARENT_DIR = pathlib.Path(__file__).parent.absolute().parent.absolute()


def main(args):
    prop = utils.get_prop(args)
    file_name_prefix = f"{PARENT_DIR}/data/raw/{prop['dataset']}/{prop['dataset']}"

    print("Data loading start...")
    X, y, prop["n_classes"] = utils.data_loader(file_name_prefix, prop["sparsity"])
    print("Data loading complete...")

    prop["seq_len"], prop["input_size"] = X.shape[1], 1
    prop["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # iterate through the stratifications
    kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    accuracies = []
    prop["kfold"] = 0
    for train_index, test_index in kf.split(X, y):
        prop["kfold"] += 1

        # batch the data points
        (
            train_data,
            train_target,
            test_data,
            test_target,
        ) = utils.preprocess(
            X[train_index], y[train_index], X[test_index], y[test_index]
        )

        print("Initializing model...")
        (
            model,
            optimizer,
            criterion_tar,
            criterion_task,
        ) = utils.initialize_model_objects(prop)
        print("Model intialized...")

        print("Model size...")
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2

        print(
            "Model size",
            param_size,
            "parameters,",
            buffer_size,
            "buffers,",
            "{:.3f}MB".format(size_all_mb),
        )

        print("Training start...")
        utils.training(
            model,
            optimizer,
            criterion_tar,
            criterion_task,
            train_data,
            train_target,
            prop,
        )
        print("Training complete...")

        print("Evaluating start...")
        metrics = utils.test(model, test_data, test_target, prop)
        print("Evaluating complete...")

        # Output the results
        accuracies.append(metrics["accuracy"])
        print(
            f"Fold: {prop['kfold']:2d}, accuracy of with {prop['model']}:  {metrics['accuracy']:4.3f}"
        )

    print(
        "**{} (epochs: {})**\n\n".format(
            file_name_prefix.split("/")[-1], prop["epochs"]
        )
    )
    print(f"\t\tMean acc.                              {np.mean(accuracies):4.3f}")
    print(f"\t\tStd. acc.                              {np.std(accuracies):4.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str, default="TARNet")
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--task_rate", type=float, default=0.5)
    parser.add_argument("--masking_ratio", type=float, default=0.15)
    parser.add_argument("--lamb", type=float, default=0.8)
    parser.add_argument("--ratio_highest_attention", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--avg", type=str, default="macro")
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nhid_task", type=int, default=128)
    parser.add_argument("--nhid_tar", type=int, default=128)
    args = parser.parse_args()
    main(args)
