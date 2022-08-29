import torch
import torch.nn as nn


class S2I_SID(nn.Module):

    def __init__(self, input_size_cell, input_size_frs, output_size = 4096):
        super(S2I_SID, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size_cell*input_size_frs, input_size_cell), nn.ReLU(),
            nn.Linear(input_size_cell, 512), nn.ReLU(),
            nn.Linear(512, output_size), nn.ReLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs).reshape(-1, 1, 64, 64)


class I2I_SID(nn.Module):

    def __init__(self):
        super(I2I_SID, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1), nn.ReLU(),
            # 这里变成4x4
            nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 3, 5, 1, 1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(3, 1, 7, 1, 1), nn.ReLU(),
            # 这里变成56x56
                                    )


    def forward(self, inputs):
        return self.layers(inputs)


class SID(nn.Module):

    def __init__(self, input_size_cell, input_size_frs, ouput_size, feature_size = 4096):
        super(SID, self).__init__()
        self.s2i = S2I_SID(input_size_cell, input_size_frs, feature_size)
        self.i2i = I2I_SID()
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(56*56, 56), nn.ReLU(),
                                 nn.Linear(56, ouput_size),
                                 )
        #接了一个mlp做映射

    def forward(self, inputs):
        return self.mlp(self.i2i(self.s2i(inputs)))

if __name__ == '__main__':
    data = torch.randn(1, 1, 16, 60)
    sid = SID(16, 60)
    y_hat = sid(data)
    print(y_hat.shape)
