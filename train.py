import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataloader.tortuosity import MyData
import numpy as np
import datetime
from prettytable import PrettyTable
from losses.label_smooth import LabelSmoothingCrossEntropyLoss
from losses.wce_loss import WeightedCrossEntropyLoss
from model.deepgrading import DeepGrading
from model.deepgrading import BiResNet
from utils.evaluation_metrics import get_metrix

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

date_now = datetime.datetime.now().strftime("-%Y-%m-%d-")

args = {
    'root'      : '/media/',
    'data_path' : 'dataset/Corneal5Folds/Fold5/',
    # 'data_path' : 'dataset/Corneal5Folds/Fold4/',
    # 'data_path' : 'dataset/Corneal5Folds/Fold3/',
    # 'data_path' : 'dataset/Corneal5Folds/Fold2/',
    # 'data_path' : 'dataset/Corneal5Folds/Fold1/',
    'epochs'    : 200,
    'lr'        : 0.001,
    'test_step' : 1,
    'ckpt_path' : 'checkpoint',
    'batch_size': 64,
    'pretrained': True,
}


def save_ckpt(net, iter):
    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    torch.save(net, os.path.join(args['ckpt_path'], 'DeepGrading-' + '.pth'))
    print('--->saved model:{}<--- '.format(args['root'] + args['ckpt_path']))


# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_acc(pred, label):
    pred = pred.data.cpu().numpy()
    label = label.data.cpu().numpy()
    right = pred == label
    right_count = len(right[right])
    return right_count / len(label)


def train():
    train_data = MyData(args['data_path'], train="train")
    batchs_data = DataLoader(train_data, batch_size=args['batch_size'], num_workers=8, shuffle=True)

    net = DeepGrading(num_class=4, model_name="resnet18", pre_train=args['pretrained'])
    # net = BiResNet(num_class=4, model_name="resnet34", pre_train=args['pretrained'])
    # net = ResNetBP(num_class=4, model_name="resnet18", pre_train=args['pretrained'])
    # net = VGG16(num_classes=4, pre_train=True)

    net = nn.DataParallel(net, device_ids=[0, 1]).cuda()
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    # set different lr in pretrained weights and the new FC layer
    if args['pretrained']:
        ignored_params = list(map(id, net.fc1.parameters()))
        ignored_params += list(map(id, net.fc2.parameters()))
        ignored_params += list(map(id, net.auxnet.parameters()))
        ignored_params += list(map(id, net.fc_bi.parameters()))
        ignored_params += list(map(id, net.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        params_list = [{'params': base_params, 'lr': args['lr']}]
        params_list.append({'params': net.fc1.parameters(), 'lr': args['lr'] * 10})
        params_list.append({'params': net.fc2.parameters(), 'lr': args['lr'] * 10})
        params_list.append({'params': net.auxnet.parameters(), 'lr': args['lr'] * 10})  # for DeepGrading
        params_list.append({'params': net.fc_bi.parameters(), 'lr': args['lr'] * 10})
        params_list.append({'params': net.classifier.parameters(), 'lr': args['lr'] * 10})
        optimizer = optim.SGD(params_list, lr=args['lr'], momentum=0.9, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(batchs_data), eta_min=1e-10)

    critrion_wce = WeightedCrossEntropyLoss().cuda()
    critrion_smooth = LabelSmoothingCrossEntropyLoss().cuda()
    print("---------------start training------------------")
    iters = 1
    best_acc1 = 0
    flag = 0
    for epoch in range(args['epochs']):
        net.train()
        epoch_num = 0
        epoch_loss = 0
        pred = []
        target = []
        for idx, batch in enumerate(batchs_data):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            image = torch.cat((img, seg, img), dim=1)
            # roi = torch.cat((roi, roi), dim=1)
            class_id = batch[1]["img_id"].cuda()
            optimizer.zero_grad()
            x1, x2, roi, predictions = net(image, roi)

            loss = critrion_smooth(predictions, class_id) + critrion_smooth(x1, class_id) + critrion_smooth(x2, class_id)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            class_id = class_id.data.cpu().numpy()
            pred.extend(predictions)
            target.extend(class_id)
            print("\t {0:d}---loss={1:8f}".format(iters, loss.item()))
            iters += 1
            epoch_num += 1
        print("Epoch {0:d}".format(epoch))
        pred = np.asarray(pred)
        target = np.asarray(target)
        wacc, wse, wsp = get_metrix(pred, target)
        scheduler.step()
        # model eval
        class_acc = model_eval(net, epoch + 1)
        if class_acc > best_acc1:
            flag = 0
            best_acc1 = class_acc
            save_ckpt(net, epoch)
            print("The current best accuracy is updated to: {}".format(best_acc1))
        else:
            flag += 1
            print("The current best accuracy is: {}".format(best_acc1))


def model_eval(net, iters):
    print("Start testing model...")
    test_data = MyData(args['data_path'], train="val")
    batchs_data = DataLoader(test_data, batch_size=args['batch_size'], num_workers=8)
    pred = []
    target = []
    net.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batchs_data):
            img = batch[0]["img"].cuda()
            seg = batch[0]["seg"].cuda()
            roi = batch[0]["roi"].cuda()
            image = torch.cat((img, seg, img), dim=1)
            # roi = torch.cat((roi, roi), dim=1)
            class_id = batch[1]["img_id"].cuda()
            x1, x2, roi, predictions = net(image, roi)
            predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            class_id = class_id.data.cpu().numpy()
            pred.extend(predictions)
            target.extend(class_id)
    pred = np.asarray(pred)
    target = np.asarray(target)
    # compute the weighted accuracy, sensitivity and specificity
    # Notion:  wacc[0]=overall wAcc, w[1]=[level1 acc, level2 acc, level3 acc, level4 acc]
    wacc, wse, wsp = get_metrix(pred, target)
    x_acc1, x_sen1, x_spe1 = iters, iters, iters
    y_acc1, y_sen1, y_spe1 = wacc[0], wse[0], wsp[0]
    table = PrettyTable()
    table.field_names = ["wAcc", "wSe", "wSp"]
    table.add_row(["{0:.4f}".format(y_acc1), "{0:.4f}".format(y_sen1), "{0:.4f}".format(y_spe1)])
    print(table)
    return wacc[0]


if __name__ == '__main__':
    train()
