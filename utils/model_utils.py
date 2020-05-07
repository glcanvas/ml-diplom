import torch
from utils import property as p
import time


def scalar(tensor):
    """
    get value from passed tensor, moved to cpu before
    :param tensor: torch Tensor
    :return: inner value from tensor
    """
    return tensor.data.cpu().item()


def reduce_to_class_number(l: int, r: int, labels, segments):
    labels = labels[:, l:r]
    segments = segments[:, l:r, :, :]
    return labels, segments


def wait_while_can_execute(model, images):
    """
    Execute the same images on model, while execute won't fail with error,
    if executed limit reached then break
    :param model: model with attention module
    :param images: passed to model images
    :return: tuple with result of model(images)
    """
    flag = True
    cnt = 0
    model_classification, model_segmentation = None, None
    while cnt != p.TRY_CALCULATE_MODEL and flag:
        try:
            cnt += 1
            model_classification, model_segmentation = model(images)
            flag = False
            torch.cuda.empty_cache()
        except RuntimeError as e:
            time.sleep(5)
            p.write_to_log("Can't execute model, CUDA out of memory", e)
    return model_classification, model_segmentation


def wait_while_can_execute_single(model, images):
    """
    Execute the same images on model, while execute won't fail with error,
    if executed limit reached then break
    :param model: model with attention module
    :param images: passed to model images
    :return: tuple with result of model(images)
    """
    flag = True
    cnt = 0
    model_classification = None
    while cnt != p.TRY_CALCULATE_MODEL and flag:
        try:
            cnt += 1
            model_classification = model(images)
            flag = False
            torch.cuda.empty_cache()
        except RuntimeError as e:
            time.sleep(5)
            p.write_to_log("Can't execute model, CUDA out of memory", e)
    return model_classification
