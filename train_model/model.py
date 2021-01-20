import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=32, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.shape[0]
        output, (hidden, _) = self.lstm(x)
        hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)
        output = self.dropout(hidden)
        output = self.linear(output)
        return output