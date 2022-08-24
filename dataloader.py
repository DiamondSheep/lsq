import os
import json
import numpy as np
import h5py
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
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        test_dataset = datasets.CIFAR10(root=path,
                                        train=False,
                                        transform=transform_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)
        return test_loader
    
    elif dataset == 'coco':
        '''
        
        class CocoDetection(data.Dataset):
        """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

        Args:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.ToTensor``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """

        def __init__(self, root, annFile, transform=None, target_transform=None):
            from pycocotools.coco import COCO
            self.root = root
            self.coco = COCO(annFile)
            self.ids = list(self.coco.imgs.keys())
            self.transform = transform
            self.target_transform = target_transform

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
            """
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

            path = coco.loadImgs(img_id)[0]['file_name']

            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target


        def __len__(self):
            return len(self.ids)

        def __repr__(self):
            fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
            fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
            fmt_str += '    Root Location: {}\n'.format(self.root)
            tmp = '    Transforms (if any): '
            fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
            tmp = '    Target Transforms (if any): '
            fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
            return fmt_str
        
        '''
        return test_loader
    else:
        return None