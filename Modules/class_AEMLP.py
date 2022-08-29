import torch.nn as nn
from copy import deepcopy
from Util.func_Util import freeze_bn


class MLP(nn.Module):
    def __init__(self, size_input, size_output):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(nn.Flatten(),
                                    nn.Linear(size_input,128),
                                    nn.ReLU(),
                                    nn.Linear(128,64),
                                    nn.ReLU(),
                                    nn.Linear(64,size_output),
                                    # nn.Softmax(dim=1)
                                    #注意softmax的用法
                                    )

    def forward(self, inputs):
        return self.layers(inputs)


class PretrainAEMLP(nn.Module):
    def __init__(self, AE, mlp):
        super(PretrainAEMLP, self).__init__()
        # 取hidden用这段代码
        self.encoder = deepcopy(AE.encoder)

        # self.encoder = deepcopy(AE)
        # 取encoder+decoder用这段

        self.freeze()
        self.mlp = deepcopy(mlp)

    def freeze(self):
        for layer in self.encoder.layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.encoder.apply(freeze_bn)
        # 仅用encoder

        # for layer in self.encoder.encoder.layers:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        # self.encoder.encoder.apply(freeze_bn)
        #
        # for layer in self.encoder.decoder.layers:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        # self.encoder.decoder.apply(freeze_bn)
        # # encoder和decoder并用
    def check_grad(self):
        for name, param in self.encoder.named_parameters():
            if param.grad:
                print(param)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        outputs = self.mlp(hidden)
        return outputs