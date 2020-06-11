import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.sleep_model import SleepNet
from models.drunk_model import DrunkNet

from models.train import trainer

use_cuda = torch.cuda.is_available()

def preprocess_sleep_data():
    transformations = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }

    train_data = torchvision.datasets.ImageFolder('src/data', transform=transformations['train'])
    valid_data = torchvision.datasets.ImageFolder('src/data', transform=transformations['val'])

    batch_size = 128
    epochs = 30
    IMG_HEIGHT = 24
    IMG_WIDTH = 24

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size)
    return train_loader, valid_loader

def run_sleep_model():
    net = SleepNet()

    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_loader, valid_loader = preprocess_data()
    train_loss, valid_loss = trainer(net, epochs, train_loader, valid_loader, optimizer, criterion, 'src/models/sleep_model.pt')

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, valid_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def preprocess_drunk_data():
    train_data = DrunkDS("src/data/drunk/train")
    test_data = DrunkDS("src/data/drunk/test")
    val_data = DrunkDS("src/data/drunk/val")

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size)

def run_drunk_model():
    net = DrunkNet()

    epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_loss, valid_loss = trainer(net, epochs, train_loader, val_loader, optimizer, criterion, 'src/models/drunk_model.pt', use_cuda)

    net.load_state_dict(torch.load('src/models/drunk_model.pt'))
    tester(test_loader, net, criterion, use_cuda)

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, valid_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
