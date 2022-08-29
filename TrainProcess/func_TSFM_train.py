import torch
import torch.nn as nn
import datetime
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Modules.class_AEMLP import PretrainAEMLP
from Util.func_Util import setup_seed, makeiter
from Modules.class_CNN import Common_CNN


def My_TSFM_Train(net, num_epochs, batch_size, lr, device, dataset_train, dataset_test, random_seed=1):
    setup_seed(random_seed)
    # 权重初始化

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
            nn.init.xavier_uniform_(m.weight)
    net.mlp.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.mlp.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss()
    # nn的交叉熵自带softmax和取负操作

    timer,  = d2l.Timer(),
    # Timer是李沐写的一个计时用的类
    now_time = datetime.datetime.now().strftime('%Y-%m-%d')
    writer = SummaryWriter(log_dir='../Summary', comment=now_time)
    record_acc = []
    for num_epoch in tqdm(range(num_epochs)):
        train_iter = makeiter(dataset_train, batch_size)
        test_iter = makeiter(dataset_test, batch_size)

        metric = d2l.Accumulator(3)
        # Accumulator也是李沐写的一个用来记录某变量累加的一个类，如此处创建了可记录
        # 三个变量的累加器，分别用来记录Sum of training loss, sum of training accuracy, no. of examples

        # 训练开始
        net.train()
        # 开启nn.Module类的训练模式
        num_batch = 0
        for i, (x, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()

            # 优化器梯度先清零
            x = x.float()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            # 优化器更新

            # 测试 batch准确率
            with torch.no_grad():
                metric.add(l * x.shape[0], d2l.accuracy(y_hat, y), x.shape[0])
            timer.stop()
            # print(l * X.shape[0],d2l.accuracy(y_hat, y),X.shape[0])
            num_batch = num_batch+1
            batch_l = metric[0] / metric[2]
            batch_acc = metric[1] / metric[2]
            writer.add_scalar('train_l', batch_l, num_batch)
            writer.add_scalar('train_acc', batch_acc, num_batch)
            # writer.add_images('train',X,num_train)
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
            # print(batch_l, batch_acc)

        # 测试epoch准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)

        # record_acc = record_acc.append(test_acc) 报错,这什么混搭用法！！代码写晕了，list、array、tensor的增删改查乱用
        record_acc.append(test_acc)
        writer.add_scalar('test_acc', test_acc, num_epoch)
        # print('epoch:', num_epoch, "\ntest_acc:", test_acc)
    # print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
    #       f'test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
    #       f'on {str(device)}')
    print("max test_acc:", max(record_acc),"num_epoch:",record_acc.index(max(record_acc)))
    writer.close()


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.
    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            if isinstance(x, list):
                # Required for BERT Fine-tuning (to be covered later)
                x = x.float()
                x = [x.to(device) for x in x]
            else:
                x = x.float()
                x = x.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(x), y), y.shape[0])

    return metric[0] / metric[1]