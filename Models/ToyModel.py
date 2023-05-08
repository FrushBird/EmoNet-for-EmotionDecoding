import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, size_output=2, size_input=96 * 64, hidden=768, cl=False):
        super(MLP, self).__init__()
        self.hidden = hidden
        if cl:
            self.layers = nn.Sequential(nn.Flatten(),
                                        # 做了Flatten，所以输入[B,C,H,W]的也可以
                                        nn.Linear(size_input, hidden),
                                        # nn.LazyLinear(hidden),
                                        nn.ReLU(),
                                        # nn.Linear(128, 64),
                                        # nn.ReLU(),
                                        # nn.Linear(64, size_output),
                                        nn.Linear(hidden, size_output),
                                        nn.Softmax(dim=-1)
                                        # 在大多数情况下，dim1和dim-1是同一行
                                        )
        else:
            self.layers = nn.Sequential(nn.Flatten(),
                                        # 做了Flatten，所以输入[B,C,H,W]的也可以
                                        nn.Linear(size_input, hidden),
                                        # nn.LazyLinear(hidden),
                                        nn.ReLU(),
                                        # nn.Linear(128, 64),
                                        # nn.ReLU(),
                                        # nn.Linear(64, size_output),
                                        nn.Linear(hidden, size_output),
                                        # 注意softmax的用法。用于CL时，启用softmax
                                        )

    def forward(self, inputs):
        return self.layers(inputs)


if __name__ == '__main__':
    data = torch.randn((100, 256), requires_grad=True)
    net = MLP(256, 3, cl=True)
    ans = net(data)
