from torch.utils.data import Dataset
import time
# import cv2
import torch
import numpy as np
from torchvision import transforms, datasets
import torch.utils.data as data
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import sys


if sys.gettrace() is not None:
    NUM_DATASET_WORKERS = 1
else:
    NUM_DATASET_WORKERS = 8

def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)




def _weights_init_cifar(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayerCifar(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayerCifar, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockCifar, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                tup1 = (0, 0, 0, 0, planes // 4, planes // 4)
                def f(x): return F.pad(x[:, :, ::2, ::2], tup1, 'constant', 0)
                self.shortcut = LambdaLayerCifar(f)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.apply(_weights_init_cifar)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet8(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']

        self.model = ResNetCifar(BasicBlockCifar, [1, 1, 1])

        self.model.conv1 = nn.Conv2d(
            in_channels,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.fc = nn.Linear(64, out_channels, bias=True)

    def forward(self, x):
        out = self.model(x)
        return out

    def __str__(self):
        model_str = str(self.model)
        return model_str

net = ResNet8({'in_channels': 3, 'out_channels': 10, 'activation': 'relu'})


transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

transform_test = transforms.Compose([
                transforms.ToTensor()])

train_data = datasets.CIFAR10('../datasets/CIFAR10/',
                              train=True,
                              download=False,
                              transform=transform_train)

test_data = datasets.CIFAR10('../datasets/CIFAR10/',
                                train=False,
                                download=False,
                                transform=transform_test)

train_loader = data.DataLoader(dataset=train_data,
                                        num_workers=NUM_DATASET_WORKERS,
                                        pin_memory=True,
                                        batch_size=128,
                                        worker_init_fn=worker_init_fn_seed,
                                        shuffle=True,
                                        drop_last=True)

test_loader = data.DataLoader(dataset=test_data,
                                        batch_size=512,
                                        shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net = net.to(device)

if __name__ == '__main__':
    best_loss = 99999
    epochs = 100
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        net.train()

        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0 
        for batch in train_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = net(imgs)
            loss = criterion(preds, labels)
            train_acc += (preds.argmax(1) == labels).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        net.eval()
        with torch.no_grad():
            test_acc = 0
            for batch in test_loader:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = net(imgs)
                test_acc += (preds.argmax(1) == labels).float().mean()
                loss = criterion(preds, labels)
                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(test_loader)
        print(f'Epoch {epoch}',
              f'Train Loss: {train_loss:.4f}',
              f'Train Acc: {train_acc / len(train_loader):.4f}',
              f'Test Loss: {valid_loss:.4f}',
              f'Test Acc: {test_acc / len(test_loader):.4f}')
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), 'best.pth')
            print('------model saved-------')





