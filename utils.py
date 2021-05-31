def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameterse()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    
    return total_parameters

def check_parameters(model):
    total_parameters = 0
    for param in model.parameters():
        n_param = param.detach().cpu().numpy()
        total_parameters += n_param

    return total_parameters