import torch
import sklearn.metrics as metrics
import property as p


def __to_global(a, b):
    aa = []
    bb = []
    for index, i in enumerate(a):
        aa.extend(list(map(lambda x: x * (index + 1), i)))
    for index, i in enumerate(b):
        bb.extend(list(map(lambda x: x * (index + 1), i)))
    return aa, bb


def scalar(tensor):
    """
    get value from passed tensor, moved to cpu before
    :param tensor: torch Tensor
    :return: inner value from tensor
    """
    return tensor.data.cpu().item()


def reduce_to_class_number(class_number, labels, segments):
    """
    if class number not none then return subset contains only this number
    :param class_number: index of class
    :param labels: bs x 5
    :param segments: bs x 5 x h x v
    :return: tuple of labels, segments
    """
    if class_number is not None:
        labels = labels[:, class_number:class_number + 1]
        segments = segments[:, class_number:class_number + 1, :, :]
        return labels, segments
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
    while cnt != p.TRY_CALCULATE_MODEL and flag:
        try:
            cnt += 1
            model_classification, model_segmentation = model(images)
            flag = False
            torch.cuda.empty_cache()
        except RuntimeError as e:
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
    while cnt != p.TRY_CALCULATE_MODEL and flag:
        try:
            cnt += 1
            model_classification = model(images)
            flag = False
            torch.cuda.empty_cache()
        except RuntimeError as e:
            p.write_to_log("Can't execute model, CUDA out of memory", e)
    return model_classification


def calculate_metric(classes, trust_answers, model_answer):
    """
    Calculate f1 score, precision, recall
    :param classes: class count
    :param trust_answers: list of list with trust answers
    :param model_answer:  list of list with model answers
    :return: tuple with f1 score, recall score, precision score
    """
    class_metric = 'binary' if classes == 1 else 'macro'

    f_1_score_text = ""
    for i in range(classes):
        f_1_score_text += "f_1_{}={:.5f} ".format(i, metrics.f1_score(trust_answers[i],
                                                                      model_answer[i], average=class_metric))
    recall_score_text = ""
    for i in range(classes):
        recall_score_text += "recall_{}={:.5f} ".format(i, metrics.recall_score(trust_answers[i],
                                                                                model_answer[i], average=class_metric))

    precision_score_text = ""
    for i in range(classes):
        precision_score_text += "precision_{}={:.5f} ".format(i, metrics.precision_score(trust_answers[i],
                                                                                         model_answer[i],
                                                                                         average=class_metric))

    trust_answer_1, model_answer_1 = __to_global(trust_answers, model_answer)
    # assert trust_answer_1 == trust_answers[0]

    f_1_score_text += "f_1_global={:.5f}".format(metrics.f1_score(trust_answer_1, model_answer_1, average=class_metric))
    recall_score_text += "recall_global={:.5f}".format(
        metrics.recall_score(trust_answer_1, model_answer_1, average=class_metric))
    precision_score_text += "precision_global={:.5f}".format(
        metrics.precision_score(trust_answer_1, model_answer_1, average=class_metric))
    return f_1_score_text, recall_score_text, precision_score_text
