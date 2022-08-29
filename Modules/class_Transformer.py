import torch
import torch.nn as nn

from Dataset.class_Dataset_Temporal import Dataset_Temp


class TempSpikeTransformer(nn.Module):
    def __init__(self):
        super(TempSpikeTransformer, self).__init__()
        self.layers = nn.Sequential(
            nn.Transformer(d_model=128, batch_first=True),
                                    )

    def forward(self, inputs, outputs):
        return self.layers(inputs, outputs)


if __name__ == '__main__':
    indexs = torch.arange(0, 100, 1)
    tempdata = Dataset_Temp(indexs)
    src, tgt, tgt_mask = tempdata[1]
    Tsf = nn.Transformer(d_model=410, nhead=10, batch_first=True,)
    # 这里可以试试用LSTM把410做成8的倍数，从而可以使用默认多头

    # embedding = nn.Embedding(10, 128)
    # outputs = Tsf(embedding(src), embedding(tgt))
    Tsf = Tsf.eval()
    outputs = Tsf(src, tgt, tgt_mask = tgt_mask)

    print(outputs.shape)