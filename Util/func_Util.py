# 公共函数区
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# Some hook may be helpful in training
def normalization(data):
    # 标准化到0-1之间
    _range = np.max(data) - np.min(data)
    if _range == 0:
        _data = data
    else:
        _data = (data - np.min(data)) / _range
    return _data


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        # os.environ['PYTHONHASHSEED'] = str(seed)


def divide_dataset(rate_train, rate_valid, path_raw_folder, path_train_folder, path_valid_folder):
    filenames = os.listdir(path_raw_folder)
    train_samples = random.sample(filenames, int(rate_train * len(filenames)))
    filenames2 = [i for i in filenames if not i in train_samples]
    valid_samples = random.sample(filenames2, int(rate_valid * len(filenames)))

    for src_file_name in train_samples:
        src_file_dir = os.path.join(path_raw_folder, src_file_name)
        des_file_dir = os.path.join(path_train_folder, src_file_name)
        shutil.copy(src_file_dir, des_file_dir)

    for src_file_name in valid_samples:
        src_file_dir = os.path.join(path_raw_folder, src_file_name)
        des_file_dir = os.path.join(path_valid_folder, src_file_name)
        shutil.copy(src_file_dir, des_file_dir)


def freeze_bn(ly):
    classname = ly.__class__.__name__
    if classname.find('BatchNorm') != -1:
        ly.eval()


def activate_bn(ly):
    classname = ly.__class__.__name__
    if classname.find('BatchNorm') != -1:
        ly.train()


def makeiter(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                             num_workers=0)  # 这里可以调成4，但我内存不够
    dataiter = iter(dataloader)
    return dataiter


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def make_weight(weights):
    scaler = StandardScaler()
    weights = scaler.fit_transform(weights.reshape(-1, 1))
    new_weights = torch.sigmoid(torch.from_numpy(weights))
    return new_weights.reshape(-1)
