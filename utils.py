import argparse

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

def get_args():
    parser = argparse.ArgumentParser('Parameter')
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10, CIFAR100, MNIST')
    parser.add_argument('--model-name',type=str, default='ResNet26', help='VGG19, GoogLeNet, ResNet26, ResNet38, ResNet50, EfficientNetB0')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    return args