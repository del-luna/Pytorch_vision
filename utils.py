import argparse
import matplotlib.pyplot as plt

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
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
    parser.add_argument('--model-name',type=str, default='VGG19', help='VGG19, GoogLeNet, ResNet18, ResNet34, ResNet50, EfficientNetB0')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers',type=int, default=0)

    args = parser.parse_args()
    return args

def visualize(train, test, mode='acc'):
    '''
    우선은 acc or loss만 받도록 구현,
    나중에 acc, loss 둘다 짜는 식으로 커스터마이징 해야됨.
    '''

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_title("Training/Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.plot(range(1, len(train)+1), train)
    ax.plot(range(1, len(test)+1), test)
    ax.legend(['Train', 'Test'])
    if mode == 'acc':
        plt.savefig(f'./visualization/model_{mode}.png')