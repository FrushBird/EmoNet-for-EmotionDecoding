import os
import re
from copy import deepcopy
import pickle

import torch

from class_Sample import FR_Sample
from Util.func_MakeReOrginSamples import MakeReOrginFRSamples

# MakeReOrginSamples -> Dataset -> Dataiter


class Dataset_FR_Train(torch.utils.data.Dataset):
    # label写在标签上哈
    def __init__(self,num_dataset):
        dir_list = [r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate\Train',
                    r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate1\Train',
                    r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate2\Train',
                    r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate3\Train'
                    ]

        self.dataset_dir = dir_list[num_dataset]

        self.labels = []
        fns = os.listdir(self.dataset_dir)
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+",fn.split('+')[1])[0])-1
            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            self.labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本,在这一步完成了raw2sample
        self.samples = deepcopy(new_samples)

    def __getitem__(self, index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_FR_Test(torch.utils.data.Dataset):

    def __init__(self, num_dataset):
        dir_list = [r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate\Test',
                    r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate1\Test',
                    r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate2\Test',
                    r'D:\WXMsWH\Warehouse\Datas\RM033dataset\FireRate3\Test'
                    ]

        self.dataset_dir = dir_list[num_dataset]
        self.labels = []
        fns = os.listdir(self.dataset_dir)
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            self.labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_FR_AE(torch.utils.data.Dataset):

    def __init__(self, indexs, num_dataset):
        dir_list = [r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate\Raw',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate1\Raw',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate2\Raw',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate3\Raw'
                    ]
        self.dataset_dir = dir_list[num_dataset]


        _ = os.listdir(self.dataset_dir)
        fns = [ _[i] for i in indexs]
        # list 只能用切片slice或者单个索引，不支持批量索引。经过这个操作后，支持K折验证

        self.labels = []
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+",fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            self.labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_FR_AEtrain(torch.utils.data.Dataset):

    def __init__(self,num_dataset):

        dir_list = [r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate\Train',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate1\Train',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate2\Train',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate3\Train'
                    ]
        self.dataset_dir = dir_list[num_dataset]
        self.labels = []
        fns = os.listdir(self.dataset_dir)
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+",fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            self.labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_FR_AEtest(torch.utils.data.Dataset):

    def __init__(self,num_dataset):
        dir_list = [r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate\Test',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate1\Test',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate2\Test',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\FireRate3\Test'
                    ]

        self.dataset_dir = dir_list[num_dataset]
        self.labels = []
        fns = os.listdir(self.dataset_dir)
        samples = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            self.labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)




if __name__ == '__main__':
    dataset_fr = Dataset_FR_Train()
    datas = torch.utils.data.DataLoader(dataset_fr, batch_size=20, shuffle=True, drop_last=False)
    # 在这台工作站上，用控制台进行多进程时会报错
    dataiter = iter(datas)
    dataiter.next()
    sample, label = dataiter.next()



