import torch
import torchvision
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader

dataset_train_first = torchvision.datasets.FashionMNIST('./', train = True, download= True,transform= torchvision.transforms.ToTensor())
dataset_test= torchvision.datasets.FashionMNIST('./', train = False, download= True,transform= torchvision.transforms.ToTensor())


print(dataset_train_first.data[0].shape)
dataset_train, dataset_val = train_test_split(dataset_train_first, test_size=0.2, random_state=4)

train_loader = DataLoader(dataset=dataset_train, batch_size= 32)
val_loader = DataLoader(dataset=dataset_val, batch_size=32)
test_loader = DataLoader(dataset=dataset_test, batch_size= 32)

print(f'length of train_loader = {train_loader}, length of val_loader = {val_loader} ' )

class Model(nn.Module):
    def __init__(self, nc) :
        super(Model, self).__init__()
        self.linear = nn.Linear(28*28, nc)

    def forward(self, x):
        x = x.view(-1, 28*28)
        y =self.linear(x)
        return y
    
model = Model(10)

optimizer = opt.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(train_loader, val_loader):
    LOSSES_train = []
    ACCS = []
    LOSSES_val = []
    for epoch in range(10):
        LOSS_train = []
        acc = 0
        for x, y in tqdm(train_loader):
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS_train.append(loss.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            acc += (torch.argmax(torch.softmax(yhat, dim=1), dim = 1) == y).float().mean()

        LOSSES_train.append(sum(LOSS_train)/len(LOSS_train))
        ACCS.append(acc)
        with torch.no_grad():
            acc = 0
            LOSS_val = []
            for x, y in tqdm(val_loader):
                yhat = model(x)
                loss = criterion(yhat, y)
                LOSS_val.append(loss.item())

                acc += (torch.argmax(torch.softmax(yhat, dim = 1), dim = 1) == y).float().mean()

            LOSSES_val.append(sum(LOSS_val))
            print(acc/len(val_loader))

    return (LOSSES_train , ACCS, LOSSES_val)

def test(test_loader):
    LOSSES_test = []
    ACCS = []
    for epoch in range(10):
        LOSS_train = []
        acc = 0
        for x, y in tqdm(test_loader):
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS_train.append(loss.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            acc += (torch.argmax(torch.softmax(yhat, dim=1), dim = 1) == y).float().mean()

        LOSSES_test.append(sum(LOSS_train)/len(LOSS_train))
        ACCS.append(acc)
        print(acc/len(test_loader))
        return (LOSSES_test , ACCS)
    

(losses_train , accs_train, losses_val) = train(train_loader, val_loader)

fg, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].set_title('LOSS TRAIN')
ax[0, 0].set_xlabel('Epoch', fontsize=6)
ax[0, 0].set_ylabel('Loss', fontsize=12)
ax[0, 0].plot(losses_train, 'ro-')

ax[0, 1].set_title('LOSS VAL')
ax[0, 1].set_xlabel('Epoch', fontsize=6)
ax[0, 1].set_ylabel('Loss', fontsize=12)
ax[0, 1].plot(losses_val, 'bo-')

ax[1, 0].set_title('Accuracy')
ax[1, 0].set_xlabel('Epoch', fontsize=6)
ax[1, 0].set_ylabel('ACC_mean', fontsize=12)
ax[1, 0].plot(accs_train, 'yo-')

plt.show()

print("Let's check accuracy the test dataset: ")
(losses_test , accs_test) = test(test_loader)

fg, ax1 = plt.subplots(1, 2, figsize=(14, 10))

ax1[0, 0].set_title('LOSS TRAIN')
ax1[0, 0].set_xlabel('Epoch', fontsize=6)
ax1[0, 0].set_ylabel('Loss', fontsize=12)
ax1[0, 0].plot(losses_test, 'ro-')

ax1[0, 1].set_title('Accuracy')
ax1[0, 1].set_xlabel('Epoch', fontsize=6)
ax1[0, 1].set_ylabel('ACC_mean', fontsize=12)
ax1[0, 1].plot(accs_test, 'bo-')

plt.show()

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(model.linear.weight[i].data.view(28,28))

with torch.no_grad():
    x, y = dataset_train[2]
    yhat = model(x)
    y = torch.argmax(torch.softmax(yhat, dim = 1), dim = 1)
    print(f't_pre = {yhat}, y_true = {y}')
    plt.imshow(x.view(28, 28), cmap = 'gray')

plt.show()

        
