import torch
from math import sqrt
from time import time
loss_tranform = lambda cost: sqrt(cost * 1)

def train(**kwargs):
    '''
    训练
    '''
    opt.parse(kwargs)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_data = VectorNetDataset(opt.train_data_root, train=True)
    val_data = VectorNetDataset(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                  collate_fn=collate)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                collate_fn=collate)

    criterion = torch.nn.MSELoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    TrainingLoss = []
    ValLoss = []
    preloss = 1e100
    for epoch in range(opt.max_epoch):
        losses = 0
        num = 0
        start_time = time()
        for ii, (data, label) in enumerate(train_dataloader):
            # input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                data['Agent'] = data['Agent'].to(torch.device('cuda:0'))
                for g in data['Map']:
                    g = g.to(torch.device('cuda:0'))
                data['Agentfeature'] = Variable(data['Agentfeature']).cuda()
                for feature in data['Mapfeature']:
                    feature = Variable(feature).cuda()
                target = target.cuda()
            if len(data['Map']) == 0:
                continue
            optimizer.zero_grad()
            score = model(data['Agent'], data['Map'], data['Agentfeature'], data['Mapfeature'], data['Mapmask'])
            loss = criterion(score.double().reshape(-1, 60), target.double())
            loss.backward()
            optimizer.step()
            losses += loss.data
            num += 1
            if num % 10 == 0:
                print(f"epoch: {epoch}, batch num: {num}, current batch loss: {loss_tranform(loss.data):.4f} m, ", \
                      f"time lapses: {time() - start_time:.1f} s")
        # model.save()
        # TrainingLoss.append(losses / num)
        TrainingLoss.append(loss_tranform(losses / num))
        # ValLoss.append(val(model, val_dataloader))
        ValLoss.append(loss_tranform(val(model, val_dataloader)))
        print(f"epoch: {epoch}, val loss: {ValLoss[-1]}, train loss: {TrainingLoss[-1]}")

        if losses / num > preloss:
            lr = lr * opt.lr_decay

        preloss = losses / num
    model.save('new.pth')


@torch.no_grad()
def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    '''
    model.eval()
    losses = 0
    num = 0
    criterion = torch.nn.MSELoss()
    for ii, (data, label) in enumerate(dataloader):
        # input = Variable(data)
        target = Variable(label)
        if opt.use_gpu:
            data['Agent'] = data['Agent'].to(torch.device('cuda:0'))
            for g in data['Map']:
                g = g.to(torch.device('cuda:0'))
            data['Agentfeature'] = data['Agentfeature'].cuda()
            for feature in data['Mapfeature']:
                feature = Variable(feature).cuda()
            target = target.cuda()
        if len(data['Map']) == 0:
            continue
        score = model(data['Agent'], data['Map'], data['Agentfeature'], data['Mapfeature'], data['Mapmask'])
        loss = criterion(score.double(), target.double())
        losses += loss.data
        num += 1
        if num % 10 == 0:
            # print(num)
            pass
    model.train()
    # print('eval:',losses/num)
    return losses / num


def test(**kwargs):
    '''
    测试（inference）
    '''
    opt.parse(kwargs)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    else:
        model.load('new.pth')
    torch.multiprocessing.set_sharing_strategy('file_system')
    test_data = VectorNetDataset(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                 collate_fn=collate)
    criterion = torch.nn.MSELoss()
    model.eval()
    losses = 0
    num = 0
    for ii, (data, label) in enumerate(test_dataloader):
        # input = Variable(data)
        target = Variable(label)
        if opt.use_gpu:
            # input = input.cuda()
            target = target.cuda()
        if len(data['Map']) == 0:
            continue
        score = model(data['Agent'], data['Map'], data['Agentfeature'], data['Mapfeature'], data['Mapmask'])
        loss = criterion(score.double(), target.double())
        print(target.flatten() - score.flatten())
        # print(score.shape)
        losses += loss.data
        num += 1
    model.train()
    print('ade:', losses / num)
    return


def help():
    '''
    打印帮助的信息 
    '''
    print('help')


if __name__ == '__main__':
    import fire
    from torch.autograd import Variable
    from models import VectorNet
    from config import DefaultConfig
    from data import VectorNetDataset, collate
    from torch.utils.data import DataLoader
    import warnings

    opt = DefaultConfig()
    model = VectorNet(5, 64, 60)
    # fire.Fire()
    warnings.filterwarnings("ignore")
    # train()
    test()
