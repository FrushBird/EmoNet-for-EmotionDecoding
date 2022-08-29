import torch
import torch.nn as nn


class BiDireLSTMEncoder(AE.Encoder):
    def __init__(self, input_size, output_size, num_hidden, num_layer, bidirect=False):
        super(BiDireLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = num_hidden

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=num_hidden,
                           num_layers=num_layer, bidirectional=bidirect)
        if bidirect:
            flag = 2
        else:
            flag = 1
        self.output_ActivateFunction = nn.Sequential(nn.ReLU(), nn.Linear(num_hidden * flag, output_size))
        # num_hidden = num_memory

    def forward(self, x, state_hidden, state_memory):
        output, state = self.rnn(x, (state_hidden, state_memory))

        # torch LSTM的输入格式要搞清楚！
        # 输出包括 output, (h_n, c_n)
        output = self.output_ActivateFunction(output)

        return output, state


class BiDireLSTMDecoder(AE.Decoder):
    def __init__(self, input_size, output_size, num_hidden, num_layer, bidirect=False):
        super(BiDireLSTMDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = num_hidden

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=num_hidden,
                           num_layers=num_layer, bidirectional=bidirect)
        # 输出格式再调整
        # if bidirect:
        #     flag = 2
        # else:
        #     flag = 1
        # self.output_ActivateFunction = nn.Sequential(nn.ReLU(), nn.Linear(num_hidden*flag, output_size))
        # # 也就是decoder输出的state

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, x, state_hidden, state_memory):
        output, state = self.rnn(x, (state_hidden, state_memory))
        # 输出包括 output, (h_n, c_n)
        output = self.output_ActivateFunction(output)
        return output, state


class BiDireLSTMEncoderDecoder(AE.EncoderDecoder):
    def __init__(self, encode, decode):
        super(BiDireLSTMEncoderDecoder, self).__init__(encoder=encode, decoder=decode)

    def forward(self, enc_x, init_hidden, init_memory):
        # input enc_X shape (nTS,Features) 注意nTS和Feature的尺寸要和encoder匹配
        if enc_x.shape[-1] != self.encoder.input_size:
            raise print('ENC输入维度有问题！')
        enc_y, (enc_state_hidden, enc_state_memory) = self.encoder(enc_x, init_hidden, init_memory)
        if enc_y.shape[-1] != self.decoder.input_size:
            raise print('DEC输入维度有问题！')

        dec_x = torch.randn_like(enc_y)
        dec_x[:] = enc_y[-1:]

        dec_y, (dec_state_hidden, dec_state_memory) = self.decoder(dec_x, enc_state_hidden, enc_state_memory)
        return dec_y, (dec_state_hidden, dec_state_memory)


if __name__ == '__main__':
    encoder = BiDireLSTMEncoder(10, 10, 20, 2, True)
    decoder = BiDireLSTMDecoder(10, 20, 20, 2, True)
    encoder_decoder = BiDireLSTMEncoderDecoder(encoder, decoder)
    inputs = torch.randn(16, 20, 10)
    init_hidden = torch.randn(4, 20, 20)
    init_memory = torch.randn(4, 20, 20)
    outputs, state = encoder_decoder(inputs, init_hidden, init_memory)
