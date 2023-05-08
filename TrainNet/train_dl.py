import sklearn
import torch
from d2l import torch as d2l
from sklearn import ensemble
from sklearn.metrics import precision_score
import numpy as np

from Util.func_Util import make_weight


def train_dl_prune(net, train_func,
                   dataset_func, train_index, test_index,
                   device=d2l.try_gpu(0), num_epochs=1000, batch_size=512,
                   lr=1e-3, rand_seed=1):
    # index输入list、numpy、tensor形式都可
    dataset_train = dataset_func(train_index)
    dataset_test = dataset_func(test_index)
    # train_CNN阔以
    train_func(net=net, num_epochs=num_epochs, batch_size=batch_size, lr=lr, device=device,
               dataset_train=dataset_train, dataset_test=dataset_test, rand_seed=rand_seed)

    print('Train Finished')
    # 2022.12.21,做了这个接口，供使用CL划分测试集之后进行训练

def train_dl_rw(net, train_func,
                dataset_func, train_index, test_index,
                device=d2l.try_gpu(0), num_dataset=1, num_epochs=1000, batch_size=512,
                lr=1e-3, rand_seed=1):
    # rw
    weight_index = make_weight(
        np.load(r'D:\WXMsWH\Warehouse\Projects\EmoNet-OS\tmp_list_rank.npy')
                               )

    # index输入list、numpy、tensor形式都可

    dataset_train = dataset_func(indexs=train_index, num_dataset=num_dataset, weight_index=weight_index)
    dataset_test = dataset_func(indexs=test_index, num_dataset=num_dataset, weight_index=weight_index)
    # train_CNN阔以
    train_func(net=net, num_epochs=num_epochs, batch_size=batch_size, lr=lr, device=device,
               dataset_train=dataset_train, dataset_test=dataset_test, rand_seed=rand_seed)

    print('Train Finished')
    # 2022.12.21,做了这个接口，供使用CL划分测试集之后进行训练

def normal_train_sk(train_index, test_index, dataset_func, num_dataset=1, rand_seed=1):
    dataset_train = dataset_func(train_index, num_dataset)
    dataset_test = dataset_func(test_index, num_dataset)

    labels_train = dataset_train.labels
    labels_test = dataset_test.labels
    samples_train = np.array([np.array(i) for i in dataset_train.samples]).reshape(len(labels_train), -1)
    samples_test = np.array([np.array(i) for i in dataset_test.samples]).reshape(len(labels_test), -1)

    # clf = svm.SVC(decision_function_shape='ovo', probability=True)
    rfc = ensemble.RandomForestClassifier(random_state=rand_seed)

    rfc.fit(samples_train, labels_train)
    # clf.fit(samples_train, labels_train)

    # y_hat_clf = clf.predict(samples_test)
    y_hat_rfc = rfc.predict(samples_test)

    acc = precision_score(labels_test, y_hat_rfc, average='macro')
    f1 = sklearn.metrics.f1_score(labels_test, y_hat_rfc, average='micro')

    print('acc_folds', acc)
    print('f1_folds', f1)
