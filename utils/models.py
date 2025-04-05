import torch.nn as nn
import torch.nn.functional as F


class FC0(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FC1(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.k*self.w*self.h, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.k*self.w*self.h)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN0(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(self.k, 3, 4, stride=(4, 4), padding='valid')
        self.conv2 = nn.Conv2d(3, 3, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc2 = nn.Linear(12, 10)
        self.m = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = self.m(x)
        x = F.relu(self.conv2(x))
        x = self.m(x)
        x = self.flatten1(x)
        x = self.fc2(x)
        return x


class CNN1(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(self.k, 6, 4, stride=(3, 3), padding='valid')
        self.conv2 = nn.Conv2d(6, 6, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(54, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN2(nn.Module):
    def __init__(self, k=1, w=28, h=28):
        super().__init__()
        self.k = k
        self.w = w
        self.h = h
        self.conv1 = nn.Conv2d(self.k, 3, 4, stride=(1, 1), padding='valid')
        self.conv2 = nn.Conv2d(3, 3, 3, stride=(3, 3), padding='valid')
        self.flatten1 = nn.Flatten()
        if self.k == 3:
            self.fc1 = nn.Linear(243, 10)
        else:
            self.fc1 = nn.Linear(192, 10)
        self.fc2 = nn.Linear(10, 10)
        self.m = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.reshape(-1, self.k, self.w, self.h)
        x = F.relu(self.conv1(x))
        x = self.m(x)
        x = F.relu(self.conv2(x))
        x = self.m(x)
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = self.fc2(x)
        return x


