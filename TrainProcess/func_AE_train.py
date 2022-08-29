import torch
import torch.nn as nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Util.func_Util import setup_seed, makeiter


def My_AE_Train(net, num_epochs, batch_size, lr, device, func_dataset_train, func_dataset_test, rand_seed=1):

    setup_seed(rand_seed)

    # 权重初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # loss = nn.SmoothL1Loss()
    loss = nn.MSELoss()
    # 选择MSE,简单粗暴

    timer = d2l.Timer()
    # Timer是李沐写的一个计时用的类

    writer = SummaryWriter(log_dir='../Summary', comment='20220706T2')
    num_train = 0
    num_data = 1
    # dataset = Dataset_FR2Pic_AEtrain
    for epoch in tqdm(range(num_epochs)):
        train_iter = makeiter(func_dataset_train, batch_size, num_data)
        metric_batch = d2l.Accumulator(2)
        # Accumulator也是李沐写的一个用来记录某变量累加的一个类，如此处创建了可记录
        # 三个变量的累加器，分别用来记录Sum of training loss, sum of training accuracy, no. of examples
        # 训练开始
        net.train()
        # 开启nn.Module类的训练模式
        for i, (x, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()

            # 优化器梯度先清零
            x = x.float()
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, x).sum()
            # 对自身训练解码器

            # l = loss(y_hat, y).sum()
            # 对标签训练

            l.backward()
            optimizer.step()
            # 优化器更新

            # 测试 batch准确率
            with torch.no_grad():
                # metric_batch.add(loss(y_hat, y).sum(), y.shape[0])
                metric_batch.add(loss(y_hat, x).sum(), x.shape[0])
            # [B,C,H,W] X.shape[0]即batch数
            timer.stop()
            # print(l * X.shape[0],d2l.accuracy(y_hat, y),X.shape[0])
            num_train = num_train + 1
            batch_l = metric_batch[0] / metric_batch[1]
            writer.add_scalar('train_l', batch_l, num_train)
            # writer.add_images('train',X,ntrain)
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
            print('Batch Train Loss:', batch_l)

        # 测试epoch的loss
        test_iter = makeiter(func_dataset_test, batch_size, num_dataset=1)
        metric_epoch = d2l.Accumulator(2)
        with torch.no_grad():
            for x, y in test_iter:
                x = x.float()
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)
                # loss_epoch = loss(y_hat, y)
                loss_epoch = loss(y_hat, x)
                # metric_epoch.add(loss_epoch, y.shape[0])
                metric_epoch.add(loss_epoch, x.shape[0])
            epoch_l = metric_epoch[0]/metric_epoch[-1]
        writer.add_scalar('epoch_l', epoch_l, epoch)
        print('Epoch Train Loss:', epoch_l)
        # print(y[0].shape, y_hat[0].shape)
        # if epoch % 10 == 0:
        #     show = ToPILImage()
        #     show(y[0]).show()
        #     show(y_hat[0]).show()

    # print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
    #       f'test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
    #       f'on {str(device)}')
    writer.close()
