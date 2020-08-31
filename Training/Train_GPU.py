import math
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
np.random.seed(1234)


def get_model(model_name, pretrained=True):
    return models.__dict__[model_name](pretrained)


class ClassificationDataset(Dataset):

    def __init__(self, imgs, targets, transform=None):
        self.imgs = imgs.float()
        self.targets = targets.long()
        self.transform = transform

        #     self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.targets[index]
        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.imgs)


def Create_Valid_data(data, target, im_size, rate=0.2, size=None):
    if size is None:
        size = len(data)

    rand_ind = np.random.randint(0, len(data), size)
    rand_valid = rand_ind[:int(size * rate)]
    rand_valid.sort()
    rand_train = rand_ind[int(size * rate):]
    rand_train.sort()
    ind = 0
    train = torch.empty((int(size * (1 - rate)) + 1, 1, im_size, im_size), dtype=torch.uint8)
    train_target = torch.empty((int(size * (1 - rate)) + 1))
    valid = torch.empty((int(size * rate), 1, im_size, im_size), dtype=torch.uint8)
    valid_target = torch.empty((int(size * rate)))
    for f in rand_train:
        train[ind, 0] = data[f, :, :]
        train_target[ind] = target[f]
        ind = ind + 1
    ind = 0
    for f in rand_valid:
        valid[ind, 0] = data[f, :, :]
        valid_target[ind] = target[f]
        ind = ind + 1

    return train, train_target, valid, valid_target

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super(SimpleCNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).cuda(device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).cuda(device)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ).cuda(device)

#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         ).cuda(device)

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # ).cuda(device)
        #
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # ).cuda(device)

        self.fc1 = nn.Linear(128 * 7 * 7, 1024).cuda(device)
        self.fc2 = nn.Linear(1024, 256).cuda(device)
        self.fc3 = nn.Linear(256, num_classes).cuda(device)

    def forward(self, x):
        x.cuda(device)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
#         out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def evaluate_model(model, dataloader):

    model.eval()
    corrects = 0
    for inputs, targets in dataloader:
        inputs = Variable(inputs)
        inputs = inputs.to(device)
        targets = Variable(targets)
        targets = targets.to(device)
        outputs = model(inputs)
        _,pred = torch.max(outputs.data,1)
        corrects += (pred == targets.data).sum()

    print('accuracy: {:.2f}'.format(100. * corrects / len(dataloader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    np.random.seed(1234)

    DATA_DIR = "D:/QuickDraw/numpy_bitmap"
    class_names = os.listdir(DATA_DIR)
    num_classes = 345
    im_size = 28
    batch_size = 32
    rate = 0.2

    r""" loading data and label(target) and calculating mean and std"""
    for i in range(num_classes):
        temp = np.load(f'{DATA_DIR}/{class_names[i]}')
        temp = torch.from_numpy(temp)
        temp = torch.reshape(temp, (-1, im_size, im_size))
        if i == 0:
            data = temp[:1250]
            target = torch.zeros((1250, 1))
        else:
            data = torch.cat([data, temp[:1250]], 0)
            target = torch.cat([target, torch.ones((1250, 1)) * i], 0)

    print(len(data))
    print(len(target))
    train, train_target, valid, valid_target = Create_Valid_data(data, target, im_size=im_size,
                                                                 rate=rate)  # TODO: this func has changed!
    del (data)
    del (target)
    print(len(train))

    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])

    train_ds = ClassificationDataset(train, train_target, transform=tfms)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    valid_ds = ClassificationDataset(valid, valid_target, transform=tfms)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    model = SimpleCNN(num_classes)
    model.cuda(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    losses = []
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_dl):
            inputs = Variable(inputs)
            inputs = inputs.to(device)
            targets = Variable(targets)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses.append(loss.item())

            loss.backward()

            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_ds) // batch_size, loss.item()))

    plt.figure(figsize=(12, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.title('Cross Entropy Loss');
    plt.show()

    evaluate_model(model, train_dl)
    print("---------------------------------------------------------------------------")
    evaluate_model(model, valid_dl)
    return model, valid_dl, valid_ds


if __name__ == '__main__':
    main()
    model,valid_dl,valid_ds = main()
    model_save_name = 'classifier-20.pt'
    path = f"D:/QuickDraw/trained models/{model_save_name}"
    torch.save(model.state_dict(), path)