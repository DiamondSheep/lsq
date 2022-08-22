import os
import lmdb
import io
import pickle
from PIL import Image
from torch.utils.data import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from .map_imagenet import label2num

def get_train_dataloader(data_path, batchsize, num_workers, distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                            ])
                            
    train_dataset = LMDBDatabase(lmdb_path=data_path, transform=transform)
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=batchsize, 
                            shuffle=(train_sampler == None), num_workers=num_workers, 
                            pin_memory=True, sampler=train_sampler)
    return train_loader

def get_val_dataloader(data_path, batchsize, num_workers, for_inception=False):
    input_size = 299 if for_inception else 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
                            transforms.Resize(int(input_size / 0.875)),
                            transforms.CenterCrop(input_size),
                            transforms.ToTensor(),
                            normalize,
                            ])
                            
    val_dataset = LMDBDatabase(lmdb_path=data_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return val_loader


class LMDBDatabase(Dataset):
    """A class for LMDB database.
    Args:
    @param lmdb: str, the source LMDB database file.
    """
    def __init__(self, lmdb_path='', transform=None):
        super().__init__()
        lmdb_file = lmdb_path
        assert isinstance(lmdb_file, str)
        print(' -- Data path: {} --'.format(lmdb_file))
        assert os.path.isfile(lmdb_file)
        self.db_binary = lmdb_file
        self.db_env = lmdb.open(lmdb_file, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.db_env.begin(write=False) as txn:
            # load meta: __len__
            db_length = txn.get(b'__len__')
            if db_length == None:
                raise ValueError('LMDB database must have __len__ item which indicates number of records in the database.')
            self.db_length = int().from_bytes(db_length, 'big') # ATTENTION! Do save __len__ as bytes in big version when making the database
            # load meta: __key__
            db_keys = txn.get(b'__keys__')
            if db_keys == None:
                raise ValueError('LMDB database must have __keys__ item which holds the keys for retrieving record.')
            self.db_keys = pickle.loads(db_keys)
            # load meta: __files__
            db_files = txn.get(b'__files__')
            if db_files == None:
                raise ValueError('LMDB database must have __files__ item which holds the paths of original data files.')
            self.db_files = pickle.loads(db_files)
        assert self.db_length == len(self.db_keys)
        self.db_iter_idx = -1
        self.transform = transform
    
    def __len__(self):
        return self.db_length
    
    def __repr__(self):
        return "%s (%s)" % (self.__class__.__name__, self.db_binary)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.db_iter_idx += 1
        if self.db_iter_idx >= len(self):
            raise StopIteration
        return self[self.db_iter_idx]
    
    def __getitem__(self, index):
        env = self.db_env
        key = self.db_keys[index]
        with env.begin(write=False) as txn:
            byteflow = txn.get(key)
            if byteflow == None: return None, None
        img = Image.open(io.BytesIO(byteflow)).convert('RGB')
        if not (self.transform == None):
            img = self.transform(img)
        label = key.decode('utf-8')
        label = label[:label.rfind('/')]
        label = label2num[label]

        return img, label

class FileDataset(Dataset):
    def __init__(self, data_path, label_file="ILSVRC2012_validation_ground_truth.txt", mode='val', transform=None):
        super().__init__()
        """
        For file folder organized in:
        data_path
            |
            label_file (.txt)
            |
            validation/
                |
                ILSVRC2012_val_00000001.JPEG
                ...
        """

        if mode == 'val':
            self.data_path = os.path.join(data_path, "validation/")
        elif mode == 'train':
            raise NotImplementedError
        else:
            self.data_path = data_path
        assert os.path.exists(self.data_path)
        print('-- data path: {} --'.format(self.data_path))
        
        self.label_file = os.path.join(data_path, label_file)
        assert os.path.exists(self.label_file)
        print('-- label file: {} --'.format(self.label_file))
        
        self.transform=transform
        self.val_idcs = None
        self.iter_idx = -1
        self.parse_val_groundtruth_txt()

    def parse_val_groundtruth_txt(self):
        with open(self.label_file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        self.val_idcs = [int(val_idx) for val_idx in val_idcs]

    def __len__(self):
        return len(self.val_idcs)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.iter_idx += 1
        if self.iter_idx >= len(self):
            raise StopIteration
        return self[self.iter_idx]
    
    def __getitem__(self, index):
        imgpath = os.path.join(self.data_path, "ILSVRC2012_val_000%05d.JPEG" % (index+1))
        img = Image.open(imgpath).convert('RGB')
        if not (self.transform == None):
            img = self.transform(img)
        label = self.val_idcs[index]
        return img, label

if __name__ == "__main__":
    pass