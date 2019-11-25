import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import image_loader as il


def to_3_dim_image(img):
    layer = img[0].tolist()
    t = torch.tensor([layer] * 3)
    t[t == 0] = 0.5
    return t


def create_empty_image(shape=(3, 224, 224)):
    t = torch.zeros(shape)
    t[t == 0] = 0.5
    return t


def visualize_cell(input_data: dict, output_data: dict):
    labels = [(i, to_3_dim_image(input_data[i])) if i in input_data else (i, create_empty_image()) for i in
              il.labels_attributes]
    labels.sort(key=lambda x: x[0])
    inputs = list(filter(lambda x: x[0] == 'input', input_data.items()))
    merged = labels + inputs
    imgs = make_grid(list(map(lambda x: x[1], merged)), padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()

    output_labels = [(i, to_3_dim_image(output_data[i])) if i in output_data else (i, create_empty_image()) for i in
                     il.labels_attributes]
    merged_labels = output_labels
    merged_labels.sort(key=lambda x: x[0])
    imgs = make_grid(list(map(lambda x: x[1], merged_labels)), padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))


def visualize_tensor(labels: torch.Tensor, inputs: torch.Tensor):
    labels = [to_3_dim_image(i) for i in labels]
    inputs = [inputs[0]]
    merged = labels + inputs
    imgs = make_grid(merged, padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()
