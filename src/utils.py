import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import os
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def accuracy_fn(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true).sum().item()
    return correct / len(y_true)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    running_acc = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_fn(outputs, y)

    return running_loss / len(loader), running_acc / len(loader)


def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0
    running_acc = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item()
            running_acc += accuracy_fn(outputs, y)

    return running_loss / len(loader), running_acc / len(loader)