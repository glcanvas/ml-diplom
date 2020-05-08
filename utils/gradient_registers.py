"""
Functions for register gradient for model builded with am_product_model.py
"""


def classifier_weights(model):
    a = list(model.classifier_branch.parameters())
    a.extend(list(model.merged_branch.parameters()))
    a.extend(list(model.avg_pool.parameters()))
    a.extend(list(model.classifier.parameters()))
    return a


def attention_module_weights(model):
    a = []
    a.extend(list(model.sam_branch.parameters()))
    return a


def register_weights(weight_class, model):
    if weight_class == "classifier":
        return classifier_weights(model)
    elif weight_class == "attention":
        return attention_module_weights(model)
    raise BaseException("unrecognized param: " + weight_class)


def disable_gradient(weight_class, model):
    if weight_class == "classifier":
        for param in classifier_weights(model):
            param.requires_grad = False
        return
    elif weight_class == "attention":
        for param in attention_module_weights(model):
            param.requires_grad = False
        return
    raise BaseException("unrecognized param: " + weight_class)


def enable_gradient(weight_class, model):
    if weight_class == "classifier":
        for param in classifier_weights(model):
            param.requires_grad = True
        return
    elif weight_class == "attention":
        for param in attention_module_weights(model):
            param.requires_grad = True
        return
    raise BaseException("unrecognized param: " + weight_class)


def enable_gradient_model(model):
    for param in model.parameters():
        param.requires_grad = True


def disable_gradient_model(model):
    for param in model.parameters():
        param.requires_grad = False
