import abc
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import *
from utils import *
from models import *

def save_checkpoint(state, is_best, filename):
    file_path = os.path.join('./checkpoint', filename) #checkpoint 파일 생성 
    torch.save(state, file_path) # state -> save to file_path
    best_file_path = os.path.join('./checkpoint', 'best_' + filename)
    if is_best:
        print('best Model Saving ...')
        shutil.copyfile(file_path, best_file_path) #filepath의 파일을 -> best_file_path로 복사

def train(model, train_loader, optimizer, criterion, args):
    model.train() #change train mode

    train_acc = 0.0
    step = 0
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad() #init gradient
        output = model(data) # return output
        loss = criterion(output, target)#calc loss
        loss.backward()#update backward
        optimizer.step()#update optimizer

        y_pred = output.data.max(1)[1] #return model predict value

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100. #calc acc
        train_acc += acc
        step += 1

    return train_acc/len(train_loader)
            
def eval(model, test_loader, args):
    print('evaluation ...')
    model.eval() #change evaluation
    correct = 0
    with torch.no_grad(): #no_grad()
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data) #return output
            prediction = output.data.max(1)[1] #return model predict value
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}'.format(acc))
    return acc

def main(args):
    '''
    load_data를 통해 data loader 정의(train, test)
    args.model_name에 저장된 model_name으로 모델 define
    criterion과 optimizer 정의

    '''
    best_acc = 0
    train_loader, test_loader = load_data(args)
    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'IMAGENET':
        pass

    if args.model_name == 'VGG19':
        model = VGG(args.model_name)
    elif args.model_name == 'Resnet18':
        model = ResNet18()
    elif args.model_name == 'Resnet34':
        model = ResNet34()
    elif args.model_name == 'Resnet50':
        model = ResNet50()
    elif args.model_name == 'EfficientNetB0':
        pass
    if args.cuda:
        print("Cuda is available!")
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained_path:
        print("load checkpoint")
        ck_point = torch.load(args.pretrained_path)
        model.load_state_dict(ck_point['state_dict'])
        optimizer.load_state_dict(ck_point['optimizer'])
        best_acc = ck_point['best_acc']
        pre_epoch = ck_point['epoch']
        train_acc_list = ck_point['train_acc_list']
        eval_acc_list = ck_point['eval_acc_list']

        for epoch in tqdm(range(pre_epoch, args.epochs + 1)):
            train_acc = train(model, train_loader, optimizer, criterion, args)
            eval_acc = eval(model, test_loader, args)
            train_acc_list.append(train_acc)
            eval_acc_list.append(eval_acc)

            is_best = eval_acc > best_acc
            best_acc = max(eval_acc, best_acc)

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = 'model_' + str(args.dataset) + '_' + str(args.model_name) + '_ckpt.tar'
            print('filename :: ', filename)

            parameters = get_model_parameters(model)

            if torch.cuda.device_count() > 1:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.model_name,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'parameters': parameters,
                    'train_acc_list':train_acc_list,
                    'eval_acc_list': eval_acc_list,
                }, is_best, filename)
            else:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.model_name,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'parameters': parameters,
                    'train_acc_list':train_acc_list,
                    'eval_acc_list': eval_acc_list,
                }, is_best, filename)

        visualize(train_acc_list, eval_acc_list, mode='acc')
    else:
        #scratch training loop
        train_acc_list = []
        eval_acc_list = []

        for epoch in tqdm(range(1, args.epochs + 1)):
            train_acc = train(model, train_loader, optimizer, criterion, args)
            #print(f'train acc: {train_acc}')
            eval_acc = eval(model, test_loader, args)
            train_acc_list.append(train_acc)
            eval_acc_list.append(eval_acc)
            is_best = eval_acc > best_acc
            best_acc = max(eval_acc, best_acc)

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = 'model_' + str(args.dataset) + '_' + str(args.model_name) + '_ckpt.tar'
            print('filename :: ', filename)

            parameters = get_model_parameters(model)

            if torch.cuda.device_count() > 1:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.model_name,
                    'state_dict': model.module.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'parameters': parameters,
                    'train_acc_list':train_acc_list,
                    'eval_acc_list': eval_acc_list,
                }, is_best, filename)
            else:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.model_name,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'parameters': parameters,
                    'train_acc_list':train_acc_list,
                    'eval_acc_list': eval_acc_list,
                }, is_best, filename)

        visualize(train_acc_list, eval_acc_list, mode='acc')

if __name__ == '__main__':

    '''
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10, CIFAR100, MNIST')
    parser.add_argument('--model-name',type=str, default='VGG19', help='VGG19, GoogLeNet, ResNet18, ResNet34, ResNet50, EfficientNetB0')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers',type=int, default=0)
    parser.add_argument('--pretrained_path',type=str, default='./checkpoint/model_CIFAR10_VGG19_ckpt.tar')
    '''
    
    best_acc, start_epoch = 0, 1
    args = get_args()
    main(args)