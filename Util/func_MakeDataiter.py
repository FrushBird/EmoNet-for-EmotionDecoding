import time
from Dataset.class_Dataset_FR import *

def makeiter_train(batch_size):
    dataset = Dataset_FR_Train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataiter = iter(dataloader)
    return dataiter

def makeiter_test(batch_size):
    dataset = Dataset_FR_Test()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataiter = iter(dataloader)
    return dataiter

def makeiter_AE(batch_size):
    dataset = Dataset_FR_AE()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataiter = iter(dataloader)
    return dataiter

def makeiter_AEtrain(batch_size):
    dataset = Dataset_FR_AEtrain()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataiter = iter(dataloader)
    return dataiter

def makeiter_AEtest(batch_size):
    dataset = Dataset_FR_AEtest()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    dataiter = iter(dataloader)
    return dataiter




def batch_ready():
    time.sleep(60)

if __name__ == '__main__':
    #测试一下
    torch_data = Dataset_FR_Train()
    datas = torch.utils.data.DataLoader(torch_data, batch_size=20, shuffle=True, drop_last=False, num_workers=2)
    dataiter = iter(datas)
    dataiter.next()
    sample,label= dataiter.next()

    sample2,label2 = makeiter_train(20).next()