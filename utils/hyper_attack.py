import argparse
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from models import *
from tqdm import tqdm


def update_attack(X, eps_pgd, alpha, size_, perturbation_type, dims):
    if perturbation_type == "brightness" or perturbation_type == "contrast":
            eps_pgd += alpha * eps_pgd.grad.sign()
            eps_pgd = torch.clamp(eps_pgd, 0, size_[0])
            eps_pgd.requires_grad = True
    elif perturbation_type == "linf" or perturbation_type == "patch":
            eps_pgd += alpha * eps_pgd.grad.sign()
            eps_pgd = torch.clamp(eps_pgd, -size_[0], size_[0])
            eps_pgd.requires_grad = True
    return eps_pgd


def create_attacked(X, eps, perturbation_type,size_,dims):
    if perturbation_type == "occ":
        row_start = int(size_[1])-1
        col_start = int(size_[2])-1
        length = int(size_[3])
        Xout = X + size_[0]
        Xout[:, :, row_start:row_start+length, col_start:col_start+length] = 0
    elif perturbation_type == "patch":
        row_start = int(size_[1])-1
        col_start = int(size_[2])-1
        length = int(size_[3])
        Xout = X + 0.0
        Xout[:, :, row_start:row_start+length, col_start:col_start+length] = torch.clamp(X[:, :, row_start:row_start+length, col_start:col_start+length]+eps,0,1)
    elif perturbation_type == "rotation":
        X = X.reshape(-1, dims[0], dims[1], dims[2])
        angle = int(size_[1])
        Xout = X - X
        height, width = dims[1],dims[2]
        center = (width // 2, height // 2)
        for i in range(height):
            for j in range(width):
                j_c = j - center[0]
                i_c = i - center[1]
                j_r = j_c * np.cos(angle * np.pi / 180) - i_c * np.sin(angle * np.pi / 180) + center[0]
                i_r = j_c * np.sin(angle * np.pi / 180) + i_c * np.cos(angle * np.pi / 180) + center[1]
                if np.floor(j_r) >= 0 and np.ceil(j_r) < width and np.floor(i_r) >= 0 and np.ceil(i_r) < height:
                    di = i_r-np.floor(i_r)
                    dj = j_r-np.floor(j_r)
                    Xout[:, :, i, j] = (1-di)*(1-dj) * X[:, :, int(np.floor(i_r)), int(np.floor(j_r))]+\
                                       di * (1-dj) * X[:, :, int(np.ceil(i_r)), int(np.floor(j_r))]+\
                                       (1-di) * dj * X[:, :, int(np.floor(i_r)), int(np.ceil(j_r))] +\
                                       di * dj * X[:, :, int(np.ceil(i_r)), int(np.ceil(j_r))]
    elif perturbation_type == "brightness":
        Xout = X+eps
    elif perturbation_type == "translation":
        m = int(size_[1])
        k = int(size_[2])
        padded_img = F.pad(X, (k, 0, m, 0), mode='constant', value=0)
        if m == 0:
            Xout = padded_img[:, :, :, :-k]
        elif k == 0:
            Xout = padded_img[:, :, :-m, :]
        else:
            Xout = padded_img[:, :, :-m, :-k]
    elif perturbation_type == "linf":
        Xout = torch.clamp(X+eps, 0, 1)
    elif perturbation_type == "filterv":
        f_coeff = [size_[1], size_[2], size_[3]]
        filter = torch.tensor(f_coeff).reshape(1, 1, len(f_coeff), 1).to('cuda:0')
        Xout = torch.nn.functional.conv2d(X, filter.view(1, 1, len(f_coeff), 1), padding=(1, 0))
    elif perturbation_type == "contrast":
        Xout = X*(1+eps)
    elif perturbation_type == "filterh":
        f_coeff = [size_[1], size_[2], size_[3]]
        filter = torch.tensor(f_coeff).reshape(1, 1, 1, len(f_coeff)).to('cuda:0')
        Xout = torch.nn.functional.conv2d(X, filter.view(1, 1, 1, len(f_coeff)), padding=(0, 1))
    else:
        assert False, "New perturbation has been detected, please add it."
    return Xout


def define_attack(perturbation_type, size_, M, dims, device):
    if perturbation_type == "patch":
        length =  int(size_[3])
        eps_pgd = torch.Tensor(M, dims[0], length, length).to(device)
        eps_pgd = eps_pgd - eps_pgd + size_[0] / 2
        eps_pgd.requires_grad = True
    elif perturbation_type == "linf":
        eps_pgd = torch.Tensor(M, dims[0], dims[1], dims[2]).to(device)
        eps_pgd = eps_pgd - eps_pgd + size_[0] / 2
        eps_pgd.requires_grad = True
    else:
        eps_pgd = torch.Tensor(M, 1, 1, 1).to(device)
        eps_pgd = eps_pgd - eps_pgd + size_[0] / 2
        eps_pgd.requires_grad = True
    return eps_pgd


def build_str(layer_data, layer_number, th=0.01):
    bools = ""
    strings = ""
    for i_c, c in enumerate(layer_data):
        if c.item() > 1 - th:
            bools += "1,"
        elif c.item() < th:
            bools += "0,"
        else:
            bools += "-1,"
        strings += "a" + str(layer_number) + "_" + str(i_c + 1) + ","
    return bools, strings


def attack(model, X, source_, target_, device, token_signature,\
           model_name, dims, type_="brightness", size_=1.0, iterations=500, alpha=0.01, lambda_0 = 1.01, K_max=500):
    model.eval()
    M = len(X)
    X_pgd = X.clone().detach()
    X_pgd.requires_grad = True
    eps_pgd = define_attack(type_, size_, M, dims, device)
    tt = target_
    ss = source_

    for t in tqdm(range(iterations), desc="Attack"):
        output = model(X_pgd)
        output2 = model(create_attacked(X_pgd,eps_pgd, type_, size_,dims))
        nan_indices = torch.isnan(output2)
        nan_rows = torch.any(nan_indices, dim=1)
        output2[nan_rows] = 0
        output_tmp = output.clone()
        output_tmp[torch.arange(M), ss] = float('-inf')
        max_not_ss, max_labels_ss = output_tmp.max(dim=1)
        diff1 = output[torch.arange(M), ss] - output[torch.arange(M), max_labels_ss]
        max_scores, max_labels = output2.max(dim=1)
        diff2 = output2[torch.arange(M), tt] - output2[torch.arange(M), max_labels]
        lambdas_ = torch.tensor((torch.absolute(diff1) / (torch.absolute(diff2) + 1e-9)).detach().cpu().numpy()).to(device)
        lambdas_.requires_grad = False
        diff = diff1 + lambda_0 * lambdas_ * diff2
        loss = torch.sum(diff)
        model.zero_grad()
        loss.backward()
        max_vals, max_inds = torch.topk(output, k=2, dim=1)
        max_labels_1 = max_inds[:, 0]
        max_vals, max_inds = torch.topk(output2, k=2, dim=1)
        max_labels_2 = max_inds[:, 0]
        s_indices = ((max_labels_1 == source_) & (max_labels_2 == target_) & (~nan_rows)).nonzero()
        if t == iterations - 1:
            break
        with torch.no_grad():
            X_pgd += alpha * X_pgd.grad.sign()
            X_pgd = torch.clamp(X_pgd, 0, 1)
            X_pgd.requires_grad = True
            eps_pgd = update_attack(X_pgd, eps_pgd, alpha, size_, type_, dims)

    k_to_use = min(K_max, len(s_indices))
    best_val = torch.Tensor([0])

    if k_to_use > 0:
        if k_to_use>1:
            s_indices = s_indices.squeeze()
        values, indices = torch.topk(diff1[s_indices], k=k_to_use)
        best_val = values[0]
        indices = s_indices[indices]
        images_to = X_pgd[indices, :]
        eps_to = eps_pgd[indices, :]
        
        layers_outputs = []
        if "FC0" in model_name or "FC1" in model_name:
            layers_outputs.append(torch.mean(torch.sign((F.relu(model.fc1(images_to.reshape(-1, 784))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(F.relu(model.fc2((F.relu(model.fc1(images_to.reshape(-1, 784))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign((F.relu(model.fc1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, 784))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(F.relu(model.fc2((F.relu(model.fc1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, 784))))))), dim=0))
        elif "CNN0" in model_name:
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2]))))),dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv2((F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2])))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign((model.flatten1(F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, dims[0], dims[1], dims[2])))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv2((F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1,  dims[0], dims[1], dims[2])))))))), dim=0))
        elif "CNN" in model_name:
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2]))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv2((F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2])))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.fc1(model.flatten1(F.relu(model.conv2((F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2]))))))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign((model.flatten1(F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, dims[0], dims[1], dims[2])))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv2((F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1,  dims[0], dims[1], dims[2])))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.fc1(model.flatten1(F.relu(model.conv2((F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1,  dims[0], dims[1], dims[2]))))))))))), dim=0))
        bools = ""
        strings = ""
        for l_no,l_data in enumerate(layers_outputs):
            b, s = build_str(l_data, l_no+1)
            bools += b
            strings += s
        bools = bools[0:-1]
        strings = strings[0:-1]
        with open("/tmp/strings_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write(strings)
        with open("/tmp/booleans_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write(bools)

    else:
        with open("/tmp/strings_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write("")
        with open("/tmp/booleans_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write("")
        with open("/tmp/fail_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write("")
    return best_val


def load_dataset(dataset):
    if dataset == "mnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)
    elif dataset == "fmnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)
    elif dataset == "cifar10":
        h_dim, w_dim, k_dim = 32, 32, 3
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
    else:
        assert False, "New dataset has been detected, please expand the code to suppourt it."


    return trainset, testset, (k_dim, h_dim, w_dim)


def load_model( model_arch, model_path, dims):
    if model_arch == "FC0":
        model = FC0(dims[0], dims[1], dims[2])
    elif model_arch == "FC1":
        model = FC1(dims[0], dims[1], dims[2])
    elif model_arch == "CNN0":
        model = CNN0(dims[0], dims[1], dims[2])
    elif model_arch == "CNN1":
        model = CNN1(dims[0], dims[1], dims[2])
    elif model_arch == "CNN2":
        model = CNN2(dims[0], dims[1], dims[2])
    else:
        assert False, "New model arch has been detected, please expand models.py and this if condition."

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def create_hyper_input(source, trainset, testset, M, dims):

    train_images = [image for image, _ in trainset]
    train_images = torch.stack(train_images).to(device)
    test_images = [image for image, _ in testset]
    test_images = torch.stack(test_images).to(device)
    random_images = torch.rand(len(trainset)+len(testset), dims[0], dims[1], dims[2]).to(device)
    all_samples = torch.cat((random_images, train_images, test_images), dim=0)
    classification = model(all_samples)
    _, predicted_labels = torch.max(classification, dim=1)
    indices_of_s = (predicted_labels == source).nonzero().squeeze()
    source_samples_classification = classification[indices_of_s]
    values, _ = torch.sort(source_samples_classification, descending=True, dim=1)
    differences = values[:, 0] - values[:, 1]
    _, sorted_indices = differences.sort(descending=True)
    sorted_indices_of_s = indices_of_s[sorted_indices]
    step_size = len(sorted_indices_of_s) // M
    uniform_indices = sorted_indices_of_s[::step_size][:M]
    hyper_input = all_samples[uniform_indices]
    return hyper_input


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VeGHar Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--source', type=float, default=0, help='source')
    parser.add_argument('--target', type=float, default=1, help='target')
    parser.add_argument('--token', type=str, default="04082021", help='token')
    parser.add_argument('--model', type=str, default="FC0", help='FC0, FC1, CNN0, CNN1 or CNN2')
    parser.add_argument('--model_path', type=str, default="./models/mnist_FC0/model.pth", help='model')
    parser.add_argument('--perturbation', type=str, default="linf", help='perturbation')
    parser.add_argument('--perturbation_size', type=str, default="1", help='perturbation size')
    parser.add_argument('--gpu', type=int, default=0, help='dataset')
    parser.add_argument('--M', type=int, default=1000, help='Number of samples to attack')
    parser.add_argument('--itr', type=int, default=500, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.01, help='Number of iterations')

    args = parser.parse_args()

    source = int(args.source)
    target = int(args.target)
    token_signature = args.token
    model_arch = args.model
    model_path = args.model_path
    perturbation_type = args.perturbation
    perturbation_size_to_parse = args.perturbation_size.split(",")
    perturbation_size = [float(i) for i in perturbation_size_to_parse]
    if perturbation_type == "occ" or perturbation_type == "translation" or perturbation_type == "rotation":
        perturbation_size = [0]+perturbation_size

    dataset = args.dataset
    M = args.M
    iterations = args.itr
    alpha = args.alpha
    if perturbation_type == "rotation": #TODO
        M = 10
        iterations = 50
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("source:", source, "target:", target, "model_arch:", model_arch, "perturbation type:", perturbation_type, \
          "perturbation size:", perturbation_size, "dataset:", dataset)

    trainset, testset, dims = load_dataset(dataset)
    model = load_model(model_arch, model_path,dims)
    X = create_hyper_input(source, trainset, testset, M, dims)

    best_val = attack(model, X, source, target, device, token_signature, model_arch, dims, perturbation_type, perturbation_size, iterations)

    with open("/tmp/best_val_" + str(source) + "_" + str(target) + "_" + str(token_signature) + ".txt", "w") as file:
        file.write(str(best_val.item()))