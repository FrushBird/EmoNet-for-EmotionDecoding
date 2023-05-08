from copy import deepcopy

from Dataset.DatasetTemplate import DatasetTemplate
from Models.ToyModel import MLP
from TrainNet.train_cl import train_cl
import numpy as np

from TrainNet.train_dl import train_dl_prune
from TrainProcess.Template import train_template


def train_EmoNet(index, dataset, net, train_epochs, num_epoch=200, batch_size=256, cl=True, dl=True):
    if cl:
        train_cl(index=index, dataset=dataset)

    if dl:
        test_index = list(np.load(r'..\tmp\tmp_list_test.npy'))
        # baseline

        print('Here is baseline:')
        net_b= deepcopy(net)
        train_index = [i for i in index if i not in np.load(r'..\tmp\tmp_list_test.npy')]
        train_dl_prune(net=net_b, train_func=train_epochs,
                       dataset_func=dataset, train_index=train_index, test_index=test_index,
                       num_epochs=num_epoch, batch_size=batch_size
                       )

        # prune
        net_p = deepcopy(net)
        train_index = [i for i in np.load(r'..\tmp\tmp_list_good.npy') if
                       i not in np.load(r'..\tmp\tmp_list_test.npy')]

        print('Here is EmoNet:')
        train_dl_prune(net=net_p, train_func=train_epochs,
                       dataset_func=dataset, train_index=train_index, test_index=test_index,
                       num_epochs=num_epoch, batch_size=batch_size
                       )


if __name__ == '__main__':

    dataset, train_epochs = DatasetTemplate, train_template
    net = MLP()
    index = list(range(0, 15705))
    train_EmoNet(index=index, dataset=dataset,
             net=net, train_epochs=train_epochs,
             cl=False, dl=True)