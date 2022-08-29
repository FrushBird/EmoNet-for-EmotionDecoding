#公共函数区
import torch
import os
import numpy as np
import random
import shutil

from Dataset.class_Dataset_FR2Pic import Dataset_FR2Pic_AEtrain

def setup_seed(seed=0):

    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        #os.environ['PYTHONHASHSEED'] = str(seed)


# 随机划分数据集的函数
def divide_dataset(rate_train, rate_valid, path_raw_folder, path_train_folder, path_valid_folder):

    filenames = os.listdir(path_raw_folder)
    # for i,filename in enumerate(filenames):
    #     filenames[i] = filename[:-4]
    # 这个有更简洁的实现吗？虽然这个功能是不需要了

    train_samples = random.sample(filenames, int(rate_train*len(filenames)))

    filenames2 =  [i for i in filenames if not i in train_samples]
    # 在挑剩下的里面选
    valid_samples = random.sample(filenames2, int(rate_valid*len(filenames)))

    for src_file_name in train_samples:
        src_file_dir = os.path.join(path_raw_folder, src_file_name)
        des_file_dir = os.path.join(path_train_folder, src_file_name)
        shutil.copy(src_file_dir, des_file_dir)

    for src_file_name in valid_samples:
        src_file_dir = os.path.join(path_raw_folder, src_file_name)
        des_file_dir = os.path.join(path_valid_folder, src_file_name)
        shutil.copy(src_file_dir, des_file_dir)


#由于BN层是由各通道值计算得出，在forward中自动实现，而不是通过梯度计算和反向传播更新，需额外冻结BN层权重
def freeze_bn(ly):
    classname = ly.__class__.__name__
    if classname.find('BatchNorm') != -1:
        ly.eval()

#用法：net.apply(freeze_bn)

# def makeiter(func_dataset, batch_size, num_dataset):
#     dataset = func_dataset(num_dataset)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
#     dataiter = iter(dataloader)
#     return dataiter

# dataset = func_dataset(num_dataset)

def makeiter(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    dataiter = iter(dataloader)
    return dataiter





if __name__ == '__main__':

    # path_RF = r'D:\RepoWxms\Mydata\RM033datasetAE\FireRate\Raw'
    # path_TR = r'D:\RepoWxms\Mydata\RM033datasetAE\FireRate\Train'
    # path_VA = r'D:\RepoWxms\Mydata\RM033datasetAE\FireRate\Test'
    # rate_train = 0.9
    # rate_valid = 0.1
    #
    # divide_dataset(rate_train, rate_valid, path_RF, path_TR, path_VA)
    dataset = Dataset_FR2Pic_AEtrain
    dataiter = makeiter(dataset, 10, 2)
    img, label = dataiter.next()
    print(img.shape)