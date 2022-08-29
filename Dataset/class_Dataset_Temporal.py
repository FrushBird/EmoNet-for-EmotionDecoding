import os
import torch
import torch.nn as nn
import re
from Util.func_MakeReOrginSamples import MakeReOrginTemporSamples
from class_Sample import *
import pickle

class Dataset_Temp(torch.utils.data.Dataset):

    def __init__(self, indexs, num_dataset = 1):
        dir_list = [
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\Temporal\Classify0\Raw',
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\Temporal\Classify1\Raw', # 3
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\Temporal\Classify2\Raw', # 2
                    r'D:\WXMsWH\Warehouse\Datas\RM033datasetAE\Temporal\Classify3\Raw',  # 3
                    ]
        self.dataset_dir = dir_list[num_dataset]


        _ = os.listdir(self.dataset_dir)
        fns = [ _[i] for i in indexs]
        # list 只能用切片slice或者单个索引，不支持批量索引。经过这个操作后，支持K折验证

        samples = []
        labels = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+", fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = Temporal_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            labels.append(ans)

        # 返回unitXTempor样本
        new_samples = MakeReOrginTemporSamples(samples)
        self.samples = deepcopy(new_samples)

        self.labels = []
        self.mask = nn.Transformer.generate_square_subsequent_mask(16)
        for i, new_sample in enumerate(new_samples):
            label = torch.zeros_like(new_sample)
            label[:, :, :] = labels[i]
            # 数据集限制，你无法打上更有效的标签

            self.labels.append(label)


    def __getitem__(self,index):
        return self.samples[index], self.labels[index], self.mask

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':

    indexs = torch.arange(0, 100, 1)
    tempdata = Dataset_Temp(indexs)
    print(tempdata[7])