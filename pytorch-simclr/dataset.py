import os
import numpy as np
import torch
import torchvision
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image


class CIFAR10Biaugment(torchvision.datasets.CIFAR10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index


class CIFAR100Biaugment(CIFAR10Biaugment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class STL10Biaugment(torchvision.datasets.STL10):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(pil_img)
            img2 = self.transform(pil_img)
        else:
            img2 = img = pil_img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index


class CIFAR10Multiaugment(torchvision.datasets.CIFAR10):

    def __init__(self, *args, n_augmentations=8, **kwargs):
        super(CIFAR10Multiaugment, self).__init__(*args, **kwargs)
        self.n_augmentations = n_augmentations
        assert self.transforms is not None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        pil_img = Image.fromarray(img)

        imgs = [self.transform(pil_img) for _ in range(self.n_augmentations)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.stack(imgs, dim=0), target, index


class CIFAR100Multiaugment(CIFAR10Multiaugment):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10Biaugment` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class ImageNetBiaugment(torchvision.datasets.ImageNet):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            img = self.transform(sample)
            img2 = self.transform(sample)
        else:
            img2 = img = sample
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, img2), target, index

def my_make_dataset(img_directory, class_file = '../preprocessed-data/abstractscenes/img_classes.npy'):
    img_classes = np.load(class_file)
    instances = list()
    for root, _, fnames in sorted(os.walk(img_directory, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if fname.endswith(".png"):
                id = fname[:-4].replace("Scene", "").replace("_", "")
                class_vector = img_classes[int(id)]
                item = path, class_vector
                instances.append(item)
    return instances
    
class AbsScenesBiaugment(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        super(AbsScenesBiaugment, self).__init__(root, transform, target_transform, **kwargs)
        
        samples = self.make_dataset(root)
        self.samples = samples
        self.imgs = self.samples
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [str(x) for x in range(0,58)]
        class_to_idx = dict(zip(classes, list(range(0,58))))
        return classes, class_to_idx
        
    def make_dataset(self, directory: str,
                     class_to_idx: Optional[Dict[str, int]] = None,
                     extensions: Optional[Union[str, Tuple[str, ...]]] = None,
                     is_valid_file: Optional[Callable[[str], bool]] = None ) -> List[Tuple[str, int]]:
        instances = my_make_dataset(directory)
        return instances
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            img = self.transform(sample)
            img2 = self.transform(sample)
        else:
            img2 = img = sample
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, img2), target, index

class AbsScenesDataset(AbsScenesBiaugment):
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        super(AbsScenesDataset, self).__init__(root, transform, target_transform, **kwargs)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            img = self.transform(sample)
        else:
            img = sample
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target 


def add_indices(dataset_cls):
    class NewClass(dataset_cls):
        def __getitem__(self, item):
            output = super(NewClass, self).__getitem__(item)
            return (*output, item)

    return NewClass



# if __name__ == '__main__':
#     # GET AbstractScenes pixel mean and std
#     import torchvision.transforms as transforms
#     from torch.utils.data import DataLoader
#     from PIL import ImageStat
#     from tqdm import tqdm
    
#     class Stats(ImageStat.Stat):
#         def __add__(self, other):
#             return Stats(list(map(np.add, self.h, other.h)))
    
#     root = '../../AbstractScenes_v1.1/RenderedScenes/'
#     transform_test = transforms.Compose([
#             transforms.Resize(224)
#         ])
#     dset = AbsScenesDataset(root=root, transform=transform_test)
    
#     statistics = None
#     for _, (img, labels, _) in tqdm(enumerate(dset)):
#         if statistics is None:
#             statistics = Stats(img)
#         else:
#             statistics += Stats(img)
#     print(f'mean:{statistics.mean}, std:{statistics.stddev}')
#     #mean:[109.10416481927726, 177.48769524248698, 134.12553284216006], std:[50.15360163453574, 45.66264343524219, 76.01617181685411]
#     #((0.428, 0.696, 0.526), (0.197, 0.179, 0.298))