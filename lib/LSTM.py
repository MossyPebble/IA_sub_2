import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    """
        Long Short-Term Memory (LSTM) 모델

        Args:
            input_size: 입력 데이터의 feature 개수
            output_size: 출력 데이터의 feature 개수
            hidden_dim: LSTM hidden state의 차원
            n_layers: LSTM의 layer 개수
    """

    def __init__(self, input_size, output_size, hidden_dim=128, n_layers=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        # 입력 데이터의 형태는 (batch_size, seq_length, input_size)이어야 하는데, (seq_length, input_size) 형태라면, 변환
        if len(x.size()) == 2:
            x = x.unsqueeze(1)

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x