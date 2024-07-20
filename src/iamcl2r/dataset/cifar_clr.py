import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

from continuum.datasets import CIFAR100
from torchvision.datasets import CIFAR10 as CIFAR10_torch

from iamcl2r.dataset.dataset_utils import subsample_dataset


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        positive_pairs =  [self.base_transform(x) for i in range(self.n_views)]
        positive_pairs = torch.stack(positive_pairs, dim=0)
        return positive_pairs

def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])
    return data_transforms




def load_cifar_clr(
    path, 
    input_size=32, 
    use_subsampled_dataset=False, 
    img_per_class=None, 
    n_views=2
):

    train_transform = [ContrastiveLearningViewGenerator(
        get_simclr_pipeline_transform(size=input_size),
        n_views=n_views
    )]

    dataset_train = CIFAR100(data_path=path, train=True, download=True)

    val_transform = [transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761))
                    ]

    dataset_val = CIFAR100(data_path=path, train=False, download=True)

    if use_subsampled_dataset:
        assert img_per_class is not None
        print(f"Subsampling dataset to {img_per_class} images per class.")
        dataset_train = subsample_dataset(dataset_train, img_per_class)
        dataset_val = subsample_dataset(dataset_val, img_per_class)
    
    return {
            "dataset_train":dataset_train, 
            "dataset_val": dataset_val, 
            "train_transform": train_transform,
            "val_transform": val_transform
            }


def load_cifar_clr_identification(path, input_size=32):

    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                        (0.2675, 0.2565, 0.2761))
                                    ])

    gallery_set = CIFAR10_torch(root=path, 
                                train=False, 
                                download=True, 
                                transform=transform
                               )
    
    query_set = CIFAR10_torch(root=path, 
                              train=True, 
                              download=True, 
                              transform=transform
                             )    
    return gallery_set, query_set

