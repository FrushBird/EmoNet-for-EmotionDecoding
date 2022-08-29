import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Util.func_MakeDataiter import *
from Modules.class_ResNet import *
from Util.func_Util import setup_seed, makeiter

def My_ResNet_Train(net, num_epochs, batch_size, lr, device, dataset_train, dataset_test):
    setup_seed(1)
    #权重初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss()
    #nn的交叉熵自带softmax和取负操作

    timer,  = d2l.Timer(),
    #Timer是李沐写的一个计时用的类

    writer = SummaryWriter(log_dir='../Summary', comment='20220706T2')
    ntrain = 0
    for epoch in tqdm(range(num_epochs)):
        train_iter = makeiter(dataset_train, batch_size)
        test_iter = makeiter(dataset_test, batch_size)
        num_batches = len(train_iter)
        metric = d2l.Accumulator(3)
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
            ntrain = ntrain+1
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            writer.add_scalar('train_l', train_l, ntrain)
            writer.add_scalar('train_acc', train_acc, ntrain)
            # print(train_l,train_acc)

        # 测试epoch准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        writer.add_scalar('test_acc', test_acc, epoch)
        print('epoch:',epoch,"\ntest_acc:",  test_acc)
    # print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
    #       f'test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
    #       f'on {str(device)}')
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
            metric.add(d2l.accuracy(net(x), y), d2l.size(y))
    return metric[0] / metric[1]

#可执行CNN、SID等训练任务

if __name__ == '__main__' :

    # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # net = vgg(conv_arch)
    net = make_ResNet18(output_size=3)
    lr, num_epochs, batch_size = 0.0001,720,24
    # 测试 BiLSTM+CNN file-single-tril分析
    setup_seed()
    My_ResNet_Train(net, num_epochs,batch_size,lr, d2l.try_gpu(0))
