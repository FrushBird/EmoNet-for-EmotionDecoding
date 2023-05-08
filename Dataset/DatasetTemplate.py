import os
import re
from copy import deepcopy
import numpy as np
import torch


class DatasetTemplate(torch.utils.data.Dataset):
    def __init__(self, indexes):
        # Please download and unzip sample datas and set up your own path

        self.dataset_dir = r'D:\WXMsWH\Warehouse\Projects\EmoNet-OS\Dataset\SampleDatas'
        _ = os.listdir(self.dataset_dir)
        _.sort(key=lambda x: int(re.findall(r"\d+", x.split('+')[0])[0]))

        fns = [_[i] for i in indexes]
        self.labels = []
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0]) - 1
            sample = torch.from_numpy(np.load(path))
            samples.append(sample)
            self.labels.append(ans)

        # [Nums,Cells,Features]
        self.samples = deepcopy(samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class DatasetTemplateForReWight(torch.utils.data.Dataset):

    def __init__(self, indexes, weight_index):
        self.dataset_dir = r'D:\WXMsWH\Warehouse\Projects\EmoNet-OS\Dataset\SampleDatas'
        _ = os.listdir(self.dataset_dir)
        _.sort(key=lambda x: int(re.findall(r"\d+", x.split('+')[0])[0]))
        fns = [_[i] for i in indexes]

        self.weight_index = [weight_index[i] for i in indexes]
        self.labels = []
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0]) - 1
            sample = torch.from_numpy(np.load(path))
            samples.append(sample)
            self.labels.append(ans)

        # [Nums,Cells,Features]
        self.samples = deepcopy(samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index], self.weight_index[index]

    def __len__(self):
        return len(self.labels)
