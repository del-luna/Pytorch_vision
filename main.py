import argparse
import os
import torch

from dataset import *

'''
main함수에 save function으로 따로 빼는게 나은지 고민해봐야함.
dataset.py만들기
Resnet을 구현해놓고 구현체를 import하는 방법 생각해보기.
근데 이러면 attention stem..을 resnet.py에 넘겨줘야하는데 어케관리하지..?

'''


def train(model, data_iter, optimizer, criterion, epoch, args):
    print('Start Training...')
    model.train()
    train_acc = 0.0

    for batch_idx (inputs, targets) in enumerate(data_iter):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, y_preds = outputs.max(1)
        acc = float(y_preds.eq(targets.data).sum()) / len(inputs) * 100.
        train_acc += acc

        if batch_idx % 10 == 0:
            print(f'epoch:[{epoch}] loss:[{loss:.3f}] acc:[{train_acc:.3f}] ') 
            

def eval(model, data_iter, args):
    print('Start Evaluation...')
    global best_acc
    model.eval()
    correct = 0

    with torch.no_grad():
        for inputs, targets in data_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inptus)
            _, y_preds = outputs.max(1)
        
        acc = 100. * float(y_preds.eq(targets.data).sum()) / len(data_iter.dataset)
        print(f'Test acc: {acc:.2f}')
        
        # Save Check point
        if acc > best_acc:
            state = {
                'model' : model.state_dict(),
                'acc' : acc,
                'epoch' : epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc



def get_args():
    parser = argparse.ArgumentParser('Parameter')
    
    parser.add_argument('--model-name',type=str, default='ResNet26', help='ResNet26, ResNet38, ResNet50')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    return args

def main(args):
    train_loader, test_loader = load_data(args)


if __name__ == '__main__':
    args = get_args()
    