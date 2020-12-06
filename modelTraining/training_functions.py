#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import random_split,DataLoader
import torch
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def prepareDatasets(train_size, validation_size, dataset):
  """Returns the train and validation datasets with the required number of items"""
  randomTrash_size = len(dataset)- train_size - validation_size
  datasets= random_split(dataset, [train_size, validation_size, randomTrash_size])
  return datasets[0], datasets[1]

def createDataLoaders(train_dataset, valid_dataset, batch_size = 100):
    """Returns the train and validation dataloaders with the required batchsize"""
    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': True}

    #Create DataLoaders
    trainDataloader = DataLoader(train_dataset, **params)
    validDataloader = DataLoader(valid_dataset, **params)

    return trainDataloader, validDataloader

def train(model, train_loader, optimizer, criterion, epoch, device, logging_freq=2):
    """Train a model.

    Arguments:
        model {torch.Module} -- The model to train.
        train_loader {dataLoader} -- dataloader with the training data.
        optimizer {Optimizer} -- Ex: Adam/SGP optimizer.
        criterion {function} -- loss function.
        epoch {int} -- number of the current epoch.
        device -- cuda or cpu.
        
    Keyword Arguments:
        logging_freq {int} -- Logging result frequency. (default: {2})
    
    """

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)/data.shape[0]
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % logging_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(),
                100. * correct / data.shape[0]))

def validation(model, val_loader, criterion, device):
    """Validate a model.

    Arguments:
        model {torch.Module} -- The model to validate.
        val_loader {dataLoader} -- dataloader with the validation data.
        criterion {function} -- loss function.
        device -- cuda or cpu.
    """
    model.eval()
    loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        # sum up batch loss
        with torch.no_grad():
            output = model(data)
            loss += criterion(output, target).data.item()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return correct / len(val_loader.dataset)

def displayTestInformation(model, testData, use_gpu = False):
    """Display model testing information:
        The label representation
        The testing accuracy
        The confusion matrix
    
    Arguments:
        model {torch.Module} -- The model to test.
        testData {dataLoader} -- dataset with the testing data.
        criterion {function} -- loss function.
        
    Keyword Arguments:
        use_gpu {bool} -- Whether to use the GPU. (default: {False})
    """
    testDataloader = DataLoader(testData, batch_size=100)
    device = torch.device(get_default_device())
    predictions =torch.empty(0)
    labels = torch.empty(0)
    if use_gpu:
      model = model.to(device)
  
    for local_batch, local_labels in testDataloader:
      if use_gpu:
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)
      response = model(local_batch)
      _, predc = response.topk(1, 1, True, True)
      if use_gpu:
        predc = predc.cpu()
        local_labels = local_labels.cpu()
      predictions = torch.cat((predictions, predc),0)
      labels= torch.cat((labels,local_labels),0)
  
    predictions = predictions.reshape(len(testData))
  
    #Display label representation
    plt.figure(figsize = (10,10)) # Label Count
    sns.set_style("darkgrid")
    sns.countplot(labels.numpy())
  
    #Display Average accuracy
    correct = predictions.t().eq(labels).to(torch.float32)
    print("Overall Testing accuracy: ", correct.mean().item())
  
    #Displaying confusion matrix
    CM=confusion_matrix(labels.numpy(),predictions.numpy()  )
    CM = (CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]*100).round()
    plt.figure(figsize = (10,10))
    sns.heatmap(CM, annot=True, xticklabels=testData.classes, yticklabels=testData.classes, cmap="Greens")
    print(CM)
