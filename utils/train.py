import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from models import *
import pickle
import argparse
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_dataset(dataset):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)
    elif dataset == "fmnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        trainset = dsets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)
    elif dataset == "cifar10":
        h_dim, w_dim, k_dim = 32, 32, 3
        trainset = dsets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
    else:
        assert False, "New dataset has been detected, please expand the code to support it."
    return trainset, testset, (k_dim, h_dim, w_dim)


def save_model(model, itr, output):
    a = []
    for i in model.parameters():
        print((np.transpose(i.cpu().detach().numpy())).shape)
        a.append(np.transpose(i.cpu().detach().numpy()))
    print("---------------", itr, "-----------------")
    for i in a:
        print(i.shape)
    model_name = "model"
    model_path = "./"+output+"/"+str(itr)+"/"
    os.system("mkdir " + model_path)
    pickle.dump(a, open(model_path + model_name + ".p", "wb"))
    torch.save(model.state_dict(), model_path + model_name + '.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--model', type=str, default="FC0", help='FC0, FC1, CNN0, CNN1 or CNN2')
    parser.add_argument('--output_dir', type=str, default="./model/", help='output directory')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--loss', type=str, default="Cross", help='Cross, MSE, or L1')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam, or SGD')
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    output_dir = args.output_dir
    model_type = args.model
    loss_type = args.loss
    optimizer_type = args.optimizer
    dataset = args.dataset

    trainset, testset, dims = load_dataset(dataset)
    if model_type == "FC0":
        model = FC0(dims[0], dims[1], dims[2])
    elif model_type == "FC1":
        model = FC1(dims[0], dims[1], dims[2])
    elif model_type == "CNN0":
        model = CNN0(dims[0], dims[1], dims[2])
    elif model_type == "CNN1":
        model = CNN1(dims[0], dims[1], dims[2])
    elif model_type == "CNN2":
        model = CNN2(dims[0], dims[1], dims[2])
    else:
        assert False, "New model arch has been detected, please expand models.py and this if condition."

    if loss_type == "Cross":
        loss = nn.CrossEntropyLoss().to(device)
    elif loss_type == "MSE":
        loss = nn.MSELoss().to(device)
    elif loss_type == "L1":
        loss = nn.L1Loss().to(device)
    else:
        assert False, "New loss has been detected, please expand this if condition to support it."

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam( model.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD( model.parameters(), lr=0.05)
    else:
        assert False, "New optimizer has been detected, please expand this if condition to support it."

    os.system("mkdir "+output_dir)
    model = model.to(device)
    print(model)

    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        total_batch = len(trainset) // batch_size
        for i, (batch_images, batch_labels) in enumerate(train_loader):

            X = batch_images.view(-1, dims[0], dims[1], dims[2]).to(device)

            Y = batch_labels.to(device)
            pre = model(X)
            if loss_type == "L1":
                Y = torch.nn.functional.one_hot(Y, 10).float()
            cost = loss(pre, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if (i + 1) % 200 == 0:
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, total_batch, cost.item()))

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images).to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum()
            print('Test accuracy: %.2f %%' % (100 * float(correct) / total))

        save_model(model, epoch, output_dir)
