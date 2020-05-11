import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()

def preprocess_data():
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 512, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def train_model(train_loader, valid_loader, optimizer, criterion, save_path):

    train_loss_log = []
    valid_loss_log = []

    valid_loss_min = np.Inf 

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        train_loss_log.append(train_loss)

        net.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        valid_loss_log.append(valid_loss)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
        ))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss
            ))
            torch.save(net.state_dict(), save_path)
            valid_loss_min = valid_loss

    print('Finished Training')
    return train_loss_log, valid_loss_log

"""
preprocess_data()
train_loss, valid_loss = train_model(train_loader, valid_loader, optimizer, criterion, 'src/models/model_test.pt')

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, valid_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""