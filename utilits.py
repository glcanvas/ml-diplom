# load images

import torch
from torchvision import datasets, transforms
import torch.utils.data.dataloader
import os


def image_data(dir_path: str, batch_size: int):
    input_size = 224

    # Data augmentation and normalization for training
    # Just normalization for validation
    composites = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # datasets.DatasetFolder(os.path.join(dir_path, 'train'), datasets.folder.default_loader, datasets.folder.IMG_EXTENSIONS)
    image_data_sets = {x: datasets.ImageFolder(os.path.join(dir_path, x), composites[x]) for x in ['train', 'val']}
    return {x: len(image_data_sets[x]) for x in ['train', 'val']}, {
        x: torch.utils.data.DataLoader(image_data_sets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x
        in ['train', 'val']}

