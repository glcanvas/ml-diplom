"""
Copy paste =(
"""

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import time


def to_3_dim_image(img):
    layer = img[0].tolist()
    t = torch.tensor([layer] * 3)
    t_min = t.min()
    t_max = t.max()
    t = (t - t_min) / (t_max - t_min)
    return t


def create_empty_image(shape=(3, 224, 224)):
    return torch.zeros(shape)


#               3                   3                   1              1
def visualize(image: torch.Tensor, i_start: torch.Tensor, a_c: torch.Tensor, mask: torch.Tensor, segment: torch.Tensor,
              epochs_trained):
    a_c = to_3_dim_image(a_c)
    # a_c[a_c > 0] = 1
    mask = to_3_dim_image(mask)
    # mask[mask > 0] = 1
    imgs = make_grid([image, i_start, to_3_dim_image(a_c), to_3_dim_image(mask), to_3_dim_image(segment)],
                     padding=50).detach().numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.figure()
    plt.show()

    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.savefig("image-I_star-A_c-mask-segment_epoch-" + str(epochs_trained) + " " + str(time.time_ns()))
    plt.figure()
