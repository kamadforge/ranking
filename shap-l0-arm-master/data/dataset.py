import torch
from torchvision import transforms, datasets
import numpy as np
import torch.utils.data as data



def mnist(batch_size=100, pm=False):
    transf = [transforms.ToTensor()]
    if pm:
        transf.append(transforms.Lambda(lambda x: x.view(-1, 784)))
    transform_data = transforms.Compose(transf)

    kwargs = {'num_workers': 0, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transform_data),
        batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform_data),
        batch_size=batch_size, shuffle=False, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar10(augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    print(logging + ' CIFAR 10.')
    kwargs = {'num_workers': 1, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=False, **kwargs)
    num_classes = 10

    return train_loader, val_loader, num_classes


def cifar100(augment=True, batch_size=128):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [129.3, 124.1, 112.4]],
                                     std=[x / 255.0 for x in [68.2, 65.4, 70.4]])

    logging = 'Using'
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        logging += ' augmented'
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    print(logging + ' CIFAR 100.')
    kwargs = {'num_workers': 1, 'pin_memory': torch.cuda.is_available()}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    num_classes = 100

    return train_loader, val_loader, num_classes

def toy_dataset(var):
    classes_num=3
    input_size=40

    x_train = np.random.multivariate_normal(np.zeros(input_size), np.eye(input_size), int(2000 / 2))
    y_train = np.random.choice(classes_num, [1000, 1])
    # x_val = np.random.multivariate_normal(np.zeros(10), np.eye(10), int(2000 / 2))

    list = []
    list_y = []
    for i in range(len(x_train)):
        list.append(x_train[i])
        list_y.append(np.array(y_train[i]))

    x_s = torch.stack([torch.Tensor(i) for i in list])
    y_s = torch.stack([torch.Tensor(i) for i in list_y])

    train_dataset = data.TensorDataset(x_s, y_s)
    train_loader = data.DataLoader(train_dataset)

    return train_loader, train_loader, classes_num



