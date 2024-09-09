import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):

    """
        단순 ANN 모델

        6개의 은닉층을 가지며, 각 층은 1000개의 노드를 가진다.

        init Args:
        
            input_size (int): 입력 데이터의 크기
            output_size (int): 출력 데이터의 크기
    """

    def __init__(self, input_size: int, output_size: int):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1000)
        self.fc5 = nn.Linear(1000, 1000)
        self.fc6 = nn.Linear(1000, 1000)
        self.fc7 = nn.Linear(1000, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x