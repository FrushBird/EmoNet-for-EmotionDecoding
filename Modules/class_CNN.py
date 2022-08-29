import torch
from torch import nn

class Common_CNN(nn.Module):

    def __init__(self,input_ch,output_size):
        super(Common_CNN, self).__init__()
        # self.layers = nn.Sequential(nn.Conv2d(input_ch, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(512, 1024, 3, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(1024, 1024, 3, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(1024, 1024, 3, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(1024, 1024, 2, 2), nn.Dropout(0.2), nn.Flatten(), nn.Linear(1024,output_size)
        #                                 # nn.Conv2d(512, 1024, 2, 2), nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.5),
        #                             )
        #   给label是3x256x256设计的

        # self.layers = nn.Sequential(nn.Conv2d(input_ch, 64, 3, (1, 2), 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(64, 128, 3, (1, 2), 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(512, 1024, 3, 2, 1), nn.BatchNorm2d(1024), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Conv2d(1024, 1024, 2, 2), nn.ReLU(), nn.Dropout(0.5),
        #                                 nn.Flatten(), nn.Linear(1024,output_size),nn.Softmax()
        #                             )
        # 给1x16x64设计的

        self.layers = nn.Sequential(nn.Conv2d(input_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Conv2d(128, 256, 2, 2), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Flatten(), nn.Linear(256, output_size),
                                        # nn.Softmax()
                                    )
        # 给256*16*16设计的, 效果在0.46、0.48、0.5 收敛更快,40多个epoch就收敛了.970多的时候到0.48-0.5

        # self.layers = nn.Sequential(nn.Conv2d(input_ch, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5),
        #                             nn.Conv2d(128, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.5),
        #                             nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout(0.5),
        #                             nn.Conv2d(64, 32, 2, 2), nn.ReLU(), nn.Dropout(0.5),
        #                             nn.Flatten(), nn.Linear(32, output_size)
                                    # nn.Softmax()
                                    # )
        # 给256*16*16设计的, 效果在0.46、0.48、0.51 100多个epoch就收敛了

    def forward(self, inputs):

        return self.layers(inputs)


class Common_MLP(nn.Module):

    def __init__(self,input_size,output_size):
        super(Common_MLP, self).__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(input_size, int(input_size/16)), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(int(input_size/16),int(input_size/256) ), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(int(input_size/256), output_size),
                                    )
        # 给1*16*16设计的

    def forward(self, inputs):

        return self.layers(inputs)




if __name__ == '__main__':
    cnn = Common_CNN(256, 3)
    mlp = Common_MLP(256*16*16, 3)
    # x = torch.randn(1, 3, 256, 256)
    # x = torch.randn(1, 1, 16, 64)
    x = torch.randn(1, 256, 16, 16)
    y = cnn(x)
    y2 = mlp(x)
    print(y.shape, y2.shape)