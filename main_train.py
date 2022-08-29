from d2l import torch as d2l
#载入数据集的过程写在train的过程里了
from sklearn.model_selection import KFold
#交叉验证


from TrainProcess.func_CNN_train import My_ResNet_Train

from Modules.class_SID import SID

from Dataset.class_Dataset_FR import *
from Util.func_Util import setup_seed

if __name__ == '__main__' :

    # net = make_ResNet18(input_channal= 3, output_size=6)
    # lr, num_epochs, batch_size = 0.001, 720, 12
    # # 测试 BiLSTM+CNN file-single-trail分析
    # setup_seed(1)
    # # 写在train里了
    # My_ResNet_Train(net, num_epochs, batch_size, lr, d2l.try_gpu(0), Dataset_Pic_train, Dataset_Pic_test)
    # # My_ResNet_Train(net, num_epochs, batch_size, lr, d2l.try_gpu(0), Dataset_PicOnlyHuman_train, Dataset_PicOnlyHuman_test)


    # setup_seed(1)
    # encoder = CNNEncoder(1, 256)
    # decoder = CNNDecoder(256, 3)
    # net = CNNEncoderDecoder(encoder, decoder)
    # lr, num_epochs, batch_size, device = 0.01, 540, 30, d2l.try_gpu(0)
    # dataset_train = Dataset_FR_AE
    # My_AE_Train(net, num_epochs, batch_size, lr, device, dataset_train, dataset_train)
    # torch.save(net, 'CNNAE_FR2_16x16Raw.pth')
    # # 训练CNNAE的代码，在全集上训练

    # encoder2 = CNNEncoder(1, 256)
    # decoder2 = CNNDecoder(256, 3)
    # net2 = net = CNNEncoderDecoder(encoder2, decoder2)
    # dataset_train = Dataset_FR_AEtrain
    # dataset_test = Dataset_FR_AEtest
    # My_AE_Train(net2, num_epochs, batch_size, lr, device, dataset_train, dataset_test)
    # torch.save(net2, 'CNNAE_FR2_16x16.pth')
    # # 训练CNNAE的代码，训练集上训练，测试集上测试

    randseed = 1
    setup_seed(randseed)

    # AE = torch.load('CNNAE_FR2_16x16Raw.pth')
    # # mlp = MLP(64 * 2 * 8, 2)
    # mlp = Common_CNN(256, 3)
    # # mlp = Common_MLP(256*16*16, 3)
    # AE_CNN = PretrainAEMLP(AE, mlp)
    # lr, num_epochs, batch_size, device = 0.001, 941 , 12, d2l.try_gpu(0)
    # 训练CNNAE+MLP的代码

    lr, num_epochs, batch_size, device = 0.001, 1000, 56, d2l.try_gpu(0)
    num_dataset = 1
    k_fold = KFold(n_splits=10, shuffle=True, random_state=randseed)
    index_list = torch.randn(600)

    for k, (train_index, test_index) in enumerate(k_fold.split(X=index_list)):

        dataset_train = Dataset_FR_AE(train_index, num_dataset)
        dataset_test = Dataset_FR_AE(test_index, num_dataset)

        # AE = torch.load('CNNAE_FR2_16x16Raw.pth')
        # # mlp = MLP(64 * 2 * 8, 2)
        # mlp = Common_CNN(256, 3)
        # # mlp = Common_MLP(256*16*16, 3)
        # AE_CNN = PretrainAEMLP(AE, mlp)
        # #载入AECNN

        # My_AEMLP_Train(AE_CNN, num_epochs, batch_size, lr, device, dataset_train, dataset_test)
        # 对于AECNN架构的测试

        sid = SID(16,64,ouput_size=3)
        My_ResNet_Train(sid, num_epochs, batch_size, lr, device, dataset_train, dataset_test)
        print('Kfold', k + 1)

    # 2022.7.31增加了K折验证。用sklearn来产生k折训练集和验证集的index；
    # 对数据集进行改造，选的时候就选指定index的；相应的，改了utils里的make dataiter函数；最后输出的时候输出最大的
    # 有待于对其他训练过程和数据集进行改造

