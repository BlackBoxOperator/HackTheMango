"""CIFAR10 example for cnn_finetune.
Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import os, sys
from cnn_finetune import make_model
from data import Mango_dataset, Eval_dataset

parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--classes', type=int, default=3, metavar='N',
                    help='number of classes (default: 3)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--finetune', type=int, default=1, metavar='S',
                    help='finetune type (default: 1)')
parser.add_argument('--augment', type=int, default=2, metavar='S',
                    help='augment type (default: 2)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, default='polynet', metavar='M',
                    help='model name (default: polynet)')
parser.add_argument('--weight-name', type=str, default='polynetw', metavar='M',
                    help='directory name for save weight (default: polynetw)')
parser.add_argument('--load', type=str, default=None, metavar='M',
                    help='path for load weight (default: None)')
parser.add_argument('--pred-csv', type=str, default=None, metavar='M',
                    help='path of csv to predict (default: None)')
parser.add_argument('--pred-dir', type=str, default=None, metavar='M',
                    help='path of directory to predict (default: None)')
parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--dataset', type=str, default='c1p1', metavar='M',
                    help='dataset (default: c1p1)')
parser.add_argument('--freeze', action='store_true',
                    help='freeze the main network')
parser.add_argument('--crop', action='store_true',
                    help='crop image by position field in dataset')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

if args.dataset == 'c1p1':
    DEV_CSV = "dev.csv"
    DEV_DIR = "C1-P1_Dev"
    TRAIN_CSV = "train.csv"
    TRAIN_DIR = "C1-P1_Train"
elif args.dataset == 'c1p2':
    DEV_CSV = "dev.csv"
    DEV_DIR = "Dev"
    TRAIN_CSV = "train.csv"
    TRAIN_DIR = "Train"
elif args.dataset == 'final':
    DEV_CSV = "new_valid0.csv"
    DEV_DIR = "Train"
    TRAIN_CSV = "new_train.csv"
    TRAIN_DIR = "Train"
else:
    print("dataset not supported yet")
    exit(0)

if args.freeze:
    FreezePretrained = True
else:
    FreezePretrained = False


def train(model, epoch, optimizer, train_loader, criterion=nn.CrossEntropyLoss()):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))


def test(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def predict(model, pred_loader, criterion=nn.CrossEntropyLoss(), classes=['A', 'B', 'C']):
    model.eval()
    test_loss = 0
    correct = 0
    df = pd.read_csv(args.pred_csv)
    with torch.no_grad():
        for data, index in pred_loader:
            data = data.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred = pred.cpu()
            df['label'][index] = classes[pred[0]]

    df.to_csv('result.csv', header=True, index=False)

def size_by_name(name, default = 256):
    #beg = name.rfind('_') + 1
    #return int(name[beg:]) if beg else default
    return default

def main(data_path=os.path.join('..', '..', args.dataset)):
    '''Main function to run code in this script'''

    model_name = args.model_name

    if model_name == 'alexnet':
        raise ValueError('The input size of the CIFAR-10 data set (32x32) is too small for AlexNet')

    make_classifier = None

    if args.finetune:
        if args.finetune == 1:
            def _make_classifier(in_features, num_classes):
                return  nn.Sequential(
                    nn.Linear(in_features, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, num_classes)
                )
            make_classifier = _make_classifier
        elif args.finetune == 2:
            def _make_classifier(in_features, num_classes):
                return  nn.Sequential(
                    nn.Linear(in_features, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, num_classes)
                )
            make_classifier = _make_classifier
        elif args.finetune == 3:
            def _make_classifier(in_features, num_classes):
                return  nn.Sequential(
                    nn.Linear(in_features, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, num_classes)
                )
            make_classifier = _make_classifier
        elif args.finetune == 4:
            def _make_classifier(in_features, num_classes):
                return  nn.Sequential(
                    nn.Linear(in_features, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, num_classes)
                )
            make_classifier = _make_classifier
        else:
            print("finetune type not supported yet")
            exit(0)

    model = make_model(
        model_name,
        pretrained=True,
        num_classes=args.classes,
        dropout_p=args.dropout_p,
        classifier_factory=make_classifier,
        input_size=(32, 32) if model_name.startswith(('vgg', 'squeezenet')) else None,
    )

    if FreezePretrained:
        for param in model.parameters():
            param.requires_grad = False 


    model = model.to(device)

    if args.augment == 0:
        transform = transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=model.original_model_info.mean,
                    std=model.original_model_info.std),
            ])
    elif args.augment == 1:
        transform = transforms.Compose([
                transforms.RandomResizedCrop(384, interpolation=2),
                transforms.RandomRotation(degrees=(-180,180)),
                transforms.ToTensor(),
                #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                transforms.Normalize(
                    mean=model.original_model_info.mean,
                    std=model.original_model_info.std),
            ])
    elif args.augment == 2:
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-180,180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((size_by_name(model_name, 224), size_by_name(model_name, 224))),
            transforms.ToTensor(),
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.Normalize(
                    mean=model.original_model_info.mean,
                    std=model.original_model_info.std),
        ])
    elif args.augment == 3:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size_by_name(model_name, 224), scale=(0.85, 1.0), interpolation=2),
            #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
            transforms.RandomRotation(degrees=(-180,180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.Normalize(
                    mean=model.original_model_info.mean,
                    std=model.original_model_info.std),
        ])
    else:
        print('augmentation type not supported yet')
        exit(0)

    print("image size is: {}".format(size_by_name(model_name, 224)))

    test_transform = transforms.Compose([
            transforms.Resize((size_by_name(model_name, 224), size_by_name(model_name, 224))),
            transforms.ToTensor(),
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.Normalize(
                    mean=model.original_model_info.mean,
                    std=model.original_model_info.std),
        ])

    print(' '.join(sys.argv))
    print(args)

    """
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    """

    train_set = Mango_dataset(
            os.path.join(data_path,TRAIN_CSV),
            os.path.join(data_path,TRAIN_DIR),
            transform, args.crop)

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    """
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, args.test_batch_size, shuffle=False, num_workers=2
    )
    """

    test_set = Mango_dataset(
            os.path.join(data_path,DEV_CSV),
            os.path.join(data_path,DEV_DIR),
            test_transform, args.crop)

    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    """
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    """

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Use exponential decay for fine-tuning optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.975)

    if args.load:
        model.load_state_dict(torch.load(args.load))
        pred_set = Eval_dataset(
            os.path.join(data_path,args.pred_csv),
            os.path.join(data_path,args.pred_dir),
            test_transform)
        pred_loader = torch.utils.data.DataLoader(
            pred_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        predict(model, pred_loader)
        return

    # Train
    w_dir = '{}w'.format(model_name) if not args.weight_name else args.weight_name
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)

    for epoch in range(1, args.epochs + 1):
        # Decay Learning Rate
        train(model, epoch, optimizer, train_loader)
        test(model, test_loader)
        scheduler.step()
        with open(os.path.join(w_dir, "weight_{}".format(epoch)), "wb") as f:
            torch.save(model.state_dict(), f)


if __name__ == '__main__':
    main()
