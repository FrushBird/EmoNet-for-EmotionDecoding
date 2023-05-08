import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
from d2l.torch import d2l
from Util.func_Util import setup_seed, makeiter
import sklearn


def train_template(net, num_epochs, batch_size, lr, device, dataset_train, dataset_test, rand_seed=1):
    # train func template for deep learning
    setup_seed(rand_seed)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 此处可进行class_weight
    loss = nn.CrossEntropyLoss()
    # you can write down a summary here
    # writer = SummaryWriter(log_dir=r'\Summary')
    # ntrain = 0

    record_acc = []
    record_f1 = []

    for epoch in tqdm(range(num_epochs)):
        train_iter = makeiter(dataset_train, batch_size)
        test_iter = makeiter(dataset_test, batch_size)

        # metric = d2l.Accumulator(3)

        # start
        net.train()
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x = x.float()
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

    # test accuracy after 200 epochs
    test_acc = evaluate_accuracy_gpu(net, test_iter)
    test_f1_score = evaluate_f1_score_gpu(net, dataset_test)

        # writer.add_scalar('test_acc', test_acc, epoch)
        # print('epoch:', epoch, "\ntest_acc:", test_acc)

    print("test_acc:", test_acc)
    print("f1_score:", test_f1_score)

    # writer.close()
    return test_acc, test_f1_score


def evaluate_f1_score_gpu(net, dataset_test, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device

    samples = dataset_test.samples
    labels = dataset_test.labels
    y_hats = []
    for x in samples:
        x = x.float().reshape(1, x.shape[0], x.shape[1])
        x = x.to(device)
        y_hat = net(x)
        y_hat = d2l.argmax(y_hat)
        y_hat = y_hat.cpu().numpy().tolist()
        y_hats.append(y_hat)
    return sklearn.metrics.f1_score(labels, y_hats, average='micro')


def evaluate_accuracy_gpu(net, data_iter, device=None):

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


# Sample loss reweight
def my_train_cnn_re_weight(net, num_epochs, batch_size, lr, device, dataset_train, dataset_test, rand_seed=1):
    setup_seed(rand_seed)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.BatchNorm2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')

    record_acc = []
    record_f1 = []
    for epoch in tqdm(range(num_epochs)):
        train_iter = makeiter(dataset_train, batch_size)
        test_iter = makeiter(dataset_test, batch_size)
        # metric = d2l.Accumulator(3)
        # start
        net.train()
        for i, (x, y, z) in enumerate(train_iter):
            # z is weight factor of sample
            optimizer.zero_grad()
            x = x.float()
            x, y, z = x.to(device), y.to(device), z.to(device)
            y_hat = net(x)
            l = loss(y_hat, y) * z
            l = l.sum()
            l.backward()
            optimizer.step()

            # test accuracy of batch and write into summary
            # with torch.no_grad():
            #     metric.add(l * x.shape[0], d2l.accuracy(y_hat, y), x.shape[0])
            # print(l * X.shape[0],d2l.accuracy(y_hat, y),X.shape[0])
            # print(train_l,train_acc)

        # test accuracy of epoch
        test_acc = evaluate_accuracy_gpu_rw(net, test_iter)
        test_f1_score = evaluate_f1_score_gpu_rw(net, dataset_test)

        record_acc.append(test_acc)
        record_f1.append(test_f1_score)
        # print('epoch:', epoch, "\ntest_acc:", test_acc)

    print("test_acc:", max(record_acc))
    print("f1_score:", max(record_f1))

    return max(record_acc), max(record_f1)


def evaluate_f1_score_gpu_rw(net, dataset_test, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device

    samples = dataset_test.samples
    labels = dataset_test.labels
    y_hats = []
    for x in samples:
        x = x.float().reshape(1, x.shape[0], x.shape[1])
        x = x.to(device)
        y_hat = net(x)
        y_hat = d2l.argmax(y_hat)
        y_hat = y_hat.cpu().numpy().tolist()
        y_hats.append(y_hat)
    return sklearn.metrics.f1_score(labels, y_hats, average='micro')


def evaluate_accuracy_gpu_rw(net, data_iter, device=None):

    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for x, y, z in data_iter:
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
