from Modules import class_AE
import torch
import torch.nn as nn


class CNNEncoder(class_AE.Encoder):
    def __init__(self, size_input, size_hidden):
        super(CNNEncoder, self).__init__()
        # self.layers = nn.Sequential(nn.Conv2d(size_input, (size_hidden * 4), 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(size_hidden * 4),
        #                             nn.Conv2d((size_hidden * 4), (size_hidden * 2), 3, 2, 1), nn.BatchNorm2d(size_hidden * 2), nn.ReLU(), nn.Dropout(0.5),
        #                             nn.Conv2d((size_hidden * 2), size_hidden, 3, 2, 1), nn.BatchNorm2d(size_hidden), nn.ReLU(), nn.Dropout(0.5))

        self.layers = nn.Sequential(nn.Conv2d(size_input, (size_hidden * 4), (3,4), (1,2), 1), nn.ReLU(),
                                    nn.BatchNorm2d(size_hidden * 4),
                                    nn.Conv2d((size_hidden * 4), (size_hidden * 2), (3,4), (1,2), 1),
                                    nn.BatchNorm2d(size_hidden * 2), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Conv2d((size_hidden * 2), size_hidden, 3, 1, 1), nn.BatchNorm2d(size_hidden),
                                    nn.ReLU(), nn.Dropout(0.5))

    # BN1D BN2D什么原理。怎么用，要考虑好
    def forward(self, inputs):
        # inputs shape (1,C,H,W)
        hidden = self.layers(inputs)
        return hidden



class CNNDecoder(class_AE.Decoder):
    def __init__(self, size_hidden, size_output):
        super(CNNDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(size_hidden, int(size_hidden /4), (3,4), (1,2), 1),
            nn.BatchNorm2d(int(size_hidden / 4)), nn.ReLU(), nn.Dropout(0.5),
            # nn.ConvTranspose2d(int(size_hidden / 2), int(size_hidden / 4), (3,4), (1,2), 1),
            nn.ConvTranspose2d(int(size_hidden / 4), size_output, (3, 4), (1, 2), 1),
            # nn.BatchNorm2d(int(size_hidden / 4)), nn.ReLU(), nn.Dropout(0.5),
            # nn.ConvTranspose2d(int(size_hidden / 4), size_output, (3,4), (1,2), 1),
            # nn.BatchNorm2d(size_output), nn.ReLU(), nn.Dropout(0.5)
            )
        # #这样输出的是1x16x64的，用于对输入自身的训练

        # self.layers = nn.Sequential(
        #     nn.ConvTranspose2d(size_hidden, int(size_hidden/2), 3, 2, 1, 1),
        #     nn.BatchNorm2d(int(size_hidden/2)), nn.ReLU(), nn.Dropout(0.5),
        #     nn.ConvTranspose2d(int(size_hidden/2), int(size_hidden/4), 3, 2, 1, 1),
        #     nn.BatchNorm2d(int(size_hidden/4)), nn.ReLU(), nn.Dropout(0.5),
        #     nn.ConvTranspose2d(int(size_hidden/4), int(size_hidden/8), 3, 2, 1, 1),
        #     nn.BatchNorm2d(int(size_hidden/8)), nn.ReLU(), nn.Dropout(0.5),
        #     nn.ConvTranspose2d(int(size_hidden/8), size_output, 3, 2, 1, 1),
        #     nn.BatchNorm2d(size_output))



    def forward(self, inputs):
        # inputs shape (1,C,H,W)
        outputs = self.layers(inputs)
        return outputs


class CNNEncoderDecoder(class_AE.EncoderDecoder):
    def __init__(self, encoder, decoder):
        super(CNNEncoderDecoder, self).__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden = None

    def hidden(self,inputs):

        return self.encoder(inputs)

    def forward(self, inputs, *args):
        self.hidden = self.encoder(inputs)
        outputs = self.decoder(self.hidden)

        return outputs


if __name__ == '__main__':
    #测试一下形状对不对
    x = torch.randn(1, 1, 16, 64)
    encoder = CNNEncoder(1, 256)
    decoder = CNNDecoder(256, 3)
    encoder_decoder = CNNEncoderDecoder(encoder,decoder)
    hidden = encoder(x)
    print('hidden', hidden.shape)
    output = decoder(hidden)
    print('output', output.shape)
    output2 = encoder_decoder(x)
    print('output2', output2.shape)
    # loss = nn.MSELoss()
    #nn里大写的东西都得先实例化再用
    # loss(x,output)
