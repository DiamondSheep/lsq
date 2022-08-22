import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from data_utils.get_imagenet import FileDataset, get_train_dataloader, get_val_dataloader

def getValData(dataset='imagenet',
                batch_size=1024,
                path='data/imagenet',
                for_inception=False):
    """
    Get dataloader of testset 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    path: the path to the data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    if dataset == 'imagenet':
        if os.path.isdir(path):
            print("-- File dataset loading...")
            input_size = 299 if for_inception else 224
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([
                                transforms.Resize(int(input_size / 0.875)),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                normalize,
                                ])
            test_dataset = FileDataset(data_path=path, 
                                    label_file="validation_gt_labels.txt", 
                                    transform=transform)
            test_loader = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=32)
        elif os.path.isfile(path):
            print("-- lmdb dataset loading...")
            test_loader = get_val_dataloader(data_path=path, batchsize=batch_size, num_workers=4, for_inception=for_inception)
        else:
            print("Wrong path")
            exit(1)
        return test_loader
        
    elif dataset == 'cifar10':
        data_dir = ' '
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        test_dataset = datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        transform=transform_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)
        return test_loader