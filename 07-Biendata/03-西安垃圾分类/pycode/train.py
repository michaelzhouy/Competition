# -*- coding: utf-8 -*-
# @Time    : 2021/9/22 9:41 下午
# @Author  : Michael Zhouy
import numpy as np
import random
import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, lr_scheduler
from PIL import ImageFile
import os
from util.prepare_data import load_data, split_data
from util.dataset import data_loader
from util.metric import eval_metric
from util.load_model import load_model
from criterion.label_smoothing import LabelSmoothSoftmaxCE
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 6
input_size = 384


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def train(data_dir, model_dir, EPOCH, model, batch_size, LR, num_workers, input_size):
    train, test = load_data(data_dir=data_dir)
    train_df, valid_df = split_data(train, test_size=2000)

    train_loader, val_loader, test_loader = data_loader(train_df, valid_df, test, batch_size=batch_size,
                                                        input_size=input_size, num_workers=num_workers)

    print('model: ', model)
    net = load_model(model, num_classes)

    loss_fn = CrossEntropyLoss()
    # optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-9)
    optimizer = SGD((net.parameters()), lr=LR, momentum=0.9, weight_decay=0.0004)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
    print('Start training...')
    for epoch in range(EPOCH):
        print('\nEPOCH: %d' % (epoch + 1))
        sum_loss, correct, total = 0.0, 0.0, 0.0
        true, pred = [], []
        net.train()
        for i, data in enumerate(train_loader):
            # optimizer.step()
            train_img, train_label = data

            train_img = train_img.to('cuda')
            train_label = torch.tensor(train_label, dtype=torch.long)
            train_label = train_label.view(-1).to('cuda')

            optimizer.zero_grad()
            output = net(train_img)
            # print('shape: ', train_label.shape, output.shape)
            train_loss = loss_fn(output, train_label)

            train_loss.backward()
            optimizer.step()

            sum_loss += train_loss.item()
            _, predicted = torch.max(output.data, 1)
            total += train_label.size(0)
            correct += predicted.eq(train_label.data).cpu().sum()
            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
            #       % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
            #          100. * float(correct) / float(total)))
            true += train_label.tolist()
            pred += predicted.tolist()
        metric = eval_metric(true, pred)
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f | Metric: %.3f'
              % (epoch + 1, (i + 1), sum_loss / (i + 1),
                 100. * float(correct) / float(total), metric))

        with torch.no_grad():
            valid_sum_loss, correct, total = 0.0, 0.0, 0.0
            true, pred = [], []
            net.eval()
            for i, data in enumerate(val_loader, 1):
                val_images, val_labels = data
                val_images = val_images.to('cuda')
                val_labels = val_labels.view(-1).to('cuda')

                output = net(val_images)
                val_loss = loss_fn(output, val_labels)

                valid_sum_loss += val_loss.item()
                _, predicted = torch.max(output.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).cpu().sum()
                true += val_labels.tolist()
                pred += predicted.tolist()
                # print('Valid acc：%.3f%%' % (100. * float(correct) / float(total)))
            metric = eval_metric(true, pred)
            print('Valid Loss: %.03f | Valid Acc: %.3f | Metric: %.3f' %
                  (valid_sum_loss / (i + 1), 100. * float(correct) / float(total), metric))
            acc = 100. * float(correct) / float(total)
            # scheduler.step(acc)

            print('Saving model...')
            print('EPOCH=%03d,Accuracy= %.3f%%' % (epoch + 1, acc))
            if not os.path.exists(os.path.join(model_dir, model)):
                os.mkdir(os.path.join(model_dir, model))
            torch.save(net.module, '%s/net_%03d.pth' % (os.path.join(model_dir, model), epoch + 1))
            save_info = {
                'optimizer': optimizer.state_dict(),
                'model': net.module.state_dict()
            }
            torch.save(save_info, '%s/params_%03d.pkl' % (os.path.join(model_dir, model), epoch + 1))

    print("Training Finished, TotalEPOCH=%d" % EPOCH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument('--data_dir', type=str, default='../data/', help='whether to img root')
    # 模型保存路径
    parser.add_argument('--model_dir', type=str, default='./model/model_pth/', help='whether to img root')
    # 迭代次数
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    # 模型
    parser.add_argument('--model', dest='model', type=str, default='resnet50', help='which net is chosen for training ')
    # 批次
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    # 图片大小
    parser.add_argument('--input_size', type=int, default=224, help='shape of each image')
    # 学习率
    parser.add_argument('--LR', type=float, default=0.01, help='LR')
    # CPU载入数据线程设置
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    # cuda设置
    parser.add_argument('--cuda', type=str, default='0,1', help='whether to use cuda if available')
    # 确认参数，并可以通过opt.xx的形式在程序中使用该参数
    opt = parser.parse_args()
    # 获取系统的cuda信息
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    print('cuda: ', opt.cuda)
    print('epochs: ', opt.epochs)
    print('batch_size: ', opt.batch_size)
    print('LR: ', opt.LR)
    set_seed(10)

    train(data_dir=opt.data_dir, model_dir=opt.model_dir, EPOCH=opt.epochs, model=opt.model, batch_size=opt.batch_size,
          LR=opt.LR, num_workers=opt.num_workers, input_size=opt.input_size)
