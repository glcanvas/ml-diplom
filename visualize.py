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
    t =torch.tensor([layer] * 3)  # flush_boundaries(torch.tensor([layer] * 3), 20)
    t_min = t.min()
    t_max = t.max()
    t = (t - t_min) / (t_max - t_min)
    return t


def flush_boundaries(t: torch.Tensor, boundaries: int):
    for chanel in t:
        for idx_i in range(0, 224):
            for idx_j in range(0, 224):
                if idx_i < boundaries or idx_i + boundaries >= 224 or idx_j < boundaries or idx_j + boundaries >= 224:
                    chanel[idx_i][idx_j] = 0
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
    plt.savefig("images/image-I_star-A_c-mask-segment_epoch-" + str(epochs_trained) + " " + str(time.time_ns()))
    plt.figure()
