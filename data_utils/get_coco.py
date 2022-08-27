import os
import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from torch.utils.data import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

class COCODataset(Dataset):
    """
        A class for COCO-2017 database.
    """
    def __init__(self, data_path='', mode='train', transform=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        if self.mode == 'train':
            self.data_path = os.path.join(self.data_path, 'train2017')
        elif self.mode == 'val':
            self.data_path = os.path.join(self.data_path, 'val2017')
        else:
            print('Unsupported dataset mode')
            raise NotImplementedError
        assert os.path.exists(self.data_path)
        print(f'-- data path: {self.data_path}')
        self.label_path = os.path.join(data_path, 'annotations')

        self.coco = COCO(os.path.join(self.label_path, f'instances_{self.mode}2017.json'))

        self.image_ids = self.coco.getImgIds()
        self.load_classes()
        self.transform = transform
    
    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.data_path, image_info['file_name'])

        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        
        img = img.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)

        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        cocos = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(cocos):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        # e.g return format
        # [[41.79 416.21 175.19 445.76  30.]
        #  [414.81 437.17 484.14 448.52  30.]]
        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        # Only for COCO
        return 80

    def __len__(self):
        return len(self.image_ids)
        
    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __iter__(self):
        return self
    
    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)

        sample = {'img': img, 'annot': annot}
        
        return sample

class mean_and_std():
    def __init__(self, loader):
        self.loader = loader
        self.sum = 0.0
        
        self.mean = torch.zeros(3)
        self.std = torch.zeros(3)

    def compute(self):
        """
        mean: [0.4694095253944397, 0.4460000991821289, 0.406586229801178], 
        std: [0.24326767027378082, 0.23839086294174194, 0.2417396754026413]
        """
        for data in self.loader:
            img = data["img"]
            for d in range(3):
                self.mean[d] += img[:, d, :, :].mean()
                self.std[d] += img[:, d, :, :].std()
        self.mean.div_(len(self.loader))
        self.std.div_(len(self.loader))
        return self.mean.tolist(), self.std.tolist()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = COCODataset(data_path = '../data/coco-2017', mode = 'val', transform = transform)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    # Computing mean and std
    ms = mean_and_std(data_loader)
    mean, std = ms.compute()
    print(f"mean: {mean}, \nstd: {std}")