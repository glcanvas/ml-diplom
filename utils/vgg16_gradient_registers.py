def base_model_weights(model):
    try:
        a = list(model.classifier_branch.parameters())
        a.extend(list(model.basis.parameters()))
        a.extend(list(model.merged_branch.parameters()))
        a.extend(list(model.avg_pool.parameters()))
        a.extend(list(model.classifier.parameters()))
        return a
    except BaseException as e:
        a = list(model.classifier_branch.parameters())
        a.extend(list(model.merged_branch.parameters()))
        a.extend(list(model.avg_pool.parameters()))
        a.extend(list(model.classifier.parameters()))
        return a


def attention_module_weights(model):
    try:
        a = []
        a.extend(list(model.basis.parameters()))
        a.extend(list(model.sam_branch.parameters()))
        return a
    except  BaseException as e:
        a = []
        a.extend(list(model.sam_branch.parameters()))
        return a


def register_weights(weight_class, model):
    if weight_class == "classifier":
        return base_model_weights(model)
    elif weight_class == "attention":
        return attention_module_weights(model)
    raise BaseException("unrecognized param: " + weight_class)
