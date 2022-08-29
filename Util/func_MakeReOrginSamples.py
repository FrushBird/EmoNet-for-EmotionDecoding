import os
from copy import deepcopy
import numpy as np
import pickle

import torch

from class_Sample import FR_Sample


def MakeReOrginFRSamples(Samples):
    #input FR_Samples_file,without depth inf
    NewSamples = []
    for Sample in Samples:
        NewSample = np.array([])
        for unit in Sample.FRs_units:
            FR = deepcopy(unit)
            # NewSample = np.append(NewSample,FR).reshape((1,-1,97))
            NewSample = np.append(NewSample, FR).reshape((1, -1, 64))
            # NewSample = NewSample.reshape((-1,80))
            #unit X ts
        NewSamples.append(NewSample)

    return NewSamples


def MakeReOrginTemporSamples(Samples):
    # input FR_Samples_file,without depth inf
    NewSamples = []
    for Sample in Samples:
        NewSample = np.array([])
        for unit in Sample.Temporal_units:
            Temp = deepcopy(unit)
            # NewSample = np.append(NewSample,FR).reshape((1,-1,97))
            NewSample = np.append(NewSample, Temp).reshape((1, -1, 410))
            # NewSample = NewSample.reshape((-1,80))
            # unit X ts
        NewSample = torch.Tensor(NewSample)
        NewSamples.append(NewSample)

    return NewSamples




if __name__ == '__main__':

    #读入样本
    dir_dataset = r'D:\RepoWxms\Mydata\RM033dataset\FireRate\Raw'
    fns = os.listdir(dir_dataset)
    Samples = []
    for fn in fns:
        path = os.path.join(dir_dataset,fn)
        with open(path, 'rb+') as f:
            Sample = FR_Sample()
            Sample = pickle.loads(f.read())
        Samples.append(Sample)

    #返回unitXFR样本
    NewSamples = MakeReOrginFRSamples(Samples)