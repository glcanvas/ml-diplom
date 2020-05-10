def base_model_weights(model):
    a = []
    a.extend(list(model.basis_branch.parameters()))
    a.extend(list(model.first_branch.parameters()))
    a.extend(list(model.merged_branch.parameters()))
    a.extend(list(model.fc.parameters()))
    return a


def attention_module_weights(model):
    a = []
    a.extend(list(model.basis_branch.parameters()))
    a.extend(list(model.am_branch.parameters()))
    return a


def register_weights(weight_class, model):
    if weight_class == "classifier":
        return base_model_weights(model)
    elif weight_class == "attention":
        return attention_module_weights(model)
    raise BaseException("unrecognized param: " + weight_class)
