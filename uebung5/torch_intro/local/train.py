import os, sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch_intro.local.model import *
from torch_intro.local.utils import *
import torch.nn.functional as F

def train(dataset, model, device, optimizer=None, criterion=None):
    """
    train() trains the model. 
    Input:
        dataset: dataset used for training.
        model: model to be trained.
        optimizer: optimizer used for training.
        criterion: defined loss function
    Return: 
        the performance on the training set and the trained model.
    """
    pass
                                                            # bring model into training mode
                                                            # traverse each batch of samples
                                                            # move data onto gpu, if gpu available

                                                            # zero the parameter gradients
                                                            # using model compute posterior probabilities
                                                            # compute loss value
                                                            # update model parameters

                                                            # compute accuracy
    return accuracy, model

def evaluation(dataset, model, device):
    """
    evaluation() is used to evaluate the model. 
    Input:
        dataset: the dataset used for evaluation.
        model: the trained model.
    Return: 
        the accuracy on the given dataset, the predictions saved in dictionary and the model.
    """
    pass
                                                            # bring model into evaluation mode
                                                            # traverse each batch of samples
                                                            # move data onto gpu if gpu available

                                                            # using trained model, compute posterior probabilities
                                                            # compute accuracy
    return accuracy, outputdict, model

def run(config, datadicts=None):
    """
    run() trains and evaluates the model over given number of epochs.
    Input:
        config: the defined hyperparameters
        datadicts: the dictionary containing the meta-data for training, dev and test set.
    """
    traindict, devdict, testdict = datadicts  
    # Parameters for feature extraction
    feat_params = [config["window_size"], config["hop_size"],
                config["feature_type"], config["n_filters"],
                config["fbank_fmin"], config["fbank_fmax"],
                config["max_frame_len"]]

    # Create 3 datasets from given training, dev and test meta-data
    train_dataset = Dataloader(traindict, feat_params)
    dev_dataset = Dataloader(devdict, feat_params)
    test_dataset = Dataloader(testdict, feat_params)

    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(test_dataset))

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    # Parameters for early stopping
    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    # Define loss function, model and optimizer
    criterion = torch.nn.BCELoss()                                                  # Binary cross entropy as loss function.
    model = Classification(idim=config["n_filters"], odim=1, hidden_dim=512)        # Initial model
    model = model.to(config["device"])                                              # move model to gpu, if gpu available
    optimizer = torch.optim.Adam(model.parameters(),                                # Initialize an optimizer
                                 lr=config["lr"]
                                 )

    # Pre-loading dataset
    data_loader_train = torch.utils.data.DataLoader(train_dataset,                  # Create dataset
                                                    shuffle=True,                   # Randomly shuffle if shuffle=True
                                                    batch_size=config["batch_size"],# Defined batch size
                                                    num_workers=config["NWORKER"],  # A positive integer will turn on multi-process data loading
                                                    drop_last=False,                # If drop_last=True, the data loader will drop the last batch if there are not enough remaining samples for a batch
                                                    collate_fn=padding)             # zero-padding is used when constructing a batch.

    data_loader_dev = torch.utils.data.DataLoader(dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=padding)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        # Train model on training set
        trainscore, model = train(data_loader_train,
                                     model,
                                     config["device"],
                                     optimizer=optimizer,
                                     criterion=criterion)
        # Evaluate trained model on dev set
        evalscore, outpre, model = evaluation(data_loader_dev, model,
                                     config["device"],)

        # Here the model is trained in one epoch, the following code saves the trained model
        # and the prediction on the development set
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']

        with open(os.path.join(resultsdir, '_'.join([str(epoch), str(currentlr), str(trainscore)[:6], str(
                              evalscore)[:6]]) + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)

        # Implementation for early stopping: If the model accuracy on the dev set does not improve in
        # 5 epochs, training is terminated.
        torch.cuda.empty_cache()
        if evalscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = evalscore
            continuescore = continuescore + 1
            OUTPUT_DIR = os.path.join(modeldir,
                          'bestaccmodel.pkl')           # save trained model
            torch.save(model, OUTPUT_DIR)

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break

    # Model has trained as many epochs as specified (subject to possible early stopping).
    # Now, evaluate the model on test set:
    data_loader_test = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=padding)

    # Load model
    model = torch.load(os.path.join(modeldir,
                          'bestaccmodel.pkl'),
                       map_location=config["device"])

    # Finally, evaluate the trained model on the test set and save the prediction.
    testacc, outpre, _ = evaluation(data_loader_test, model, config["device"])
    with open(os.path.join(resultsdir,
                           'testacc_' + str(round(testacc, 6)) + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)





