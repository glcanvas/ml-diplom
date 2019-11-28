import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import property as P


def to_3_dim_image(img):
    layer = img[0].tolist()
    return torch.tensor([layer] * 3)


def create_empty_image(shape=(3, 224, 224)):
    return torch.zeros(shape)


# first try
def visualize_cell(input_data: dict, output_data: dict):
    labels = [(i, to_3_dim_image(input_data[i])) if i in input_data else (i, create_empty_image()) for i in
              P.labels_attributes]
    labels.sort(key=lambda x: x[0])
    inputs = list(filter(lambda x: x[0] == 'input', input_data.items()))
    merged = labels + inputs
    imgs = make_grid(list(map(lambda x: x[1], merged)), padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()

    output_labels = [(i, to_3_dim_image(output_data[i])) if i in output_data else (i, create_empty_image()) for i in
                     P.labels_attributes]
    merged_labels = output_labels
    merged_labels.sort(key=lambda x: x[0])
    imgs = make_grid(list(map(lambda x: x[1], merged_labels)), padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))


# second try
def visualize_tensor(labels: torch.Tensor, inputs: torch.Tensor):
    labels = [to_3_dim_image(i) for i in labels]
    inputs = [inputs]
    merged = labels + inputs
    imgs = make_grid(merged, padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()


def visualize_single_tensor(label: torch.Tensor, inputs: torch.Tensor):
    labels = [to_3_dim_image(label)]
    inputs = [inputs]
    merged = labels + inputs
    imgs = make_grid(merged, padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()


# third try
def visualize_colorized_tensor(label: torch.Tensor, inputs: torch.Tensor):
    """
    draw grid
    :param label: 1 x 224 x 224 weighted images
    :param inputs: 3 x 224 x 224 rgb image
    :return:
    """
    labels = [torch.zeros((1, 224, 224)) for _ in range(len(P.labels_attributes))]
    for idx, i in enumerate(P.labels_attributes_weight):
        # colorize model
        label[abs(label - i) < i / 2] = i
        labels[idx] = torch.where(label == i, label, labels[idx])
        # mark visited color as 0
        label[label == i] = 0
        # non zeros pixels to 1
        tmp = labels[idx]
        tmp[tmp > 0] = 1

    labels = list(map(lambda x: to_3_dim_image(x), labels))
    inputs = [inputs]
    merged = labels + inputs
    imgs = make_grid(merged, padding=50).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()
