import torch
import torch.nn as nn
from config import ConfigLSTM


class VAD_LSTM(nn.Module):
    def __init__(self, conf):
        super(VAD_LSTM, self).__init__()
        self.seq_size = conf.seq_size
        self.hidden_size = conf.hidden_size
        self.num_layers = conf.num_layers
        self.dropout_p = conf.dropout_p
        self.output_size = conf.output_size
        self.input_size = conf.input_size
        self.num_delay = conf.num_delay
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * self.hidden_size, self.output_size)
        self.drop = nn.Dropout(self.dropout_p)

    def rnn_input(self, x):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        len_input = x.shape[0]
        assert len_input % self.seq_size == 0
        batch_size = len_input // self.seq_size
        x = torch.cat((x, torch.zeros((self.num_delay, self.input_size)).to(device)))
        temp1 = torch.reshape(x, ([-1, self.input_size]))
        temp2 = torch.reshape(temp1[: len_input], ([batch_size, self.seq_size, -1]))
        temp3 = temp2[1: batch_size, : self.num_delay]
        temp4 = torch.reshape(temp1[len_input:], ([1, self.num_delay, -1]))
        temp5 = torch.cat((temp3, temp4), 0)
        return torch.cat((temp2, temp5), 1)

    def forward(self, X):
        input_rnn = self.rnn_input(X)
        output, _ = self.rnn(input_rnn)
        output = torch.reshape(output[:, : self.seq_size], [-1, self.hidden_size])
        output = self.drop(output)
        output = output.view(output.shape[0]//2, output.shape[1]*2)
        output = self.fc(output)
        return output