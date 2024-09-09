import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    """
        Convolutional Neural Network 모델

        2개의 Convolutional Layer와 2개의 Fully Connected Layer로 구성된 모델

        Args:
            input_size: 입력 데이터의 feature 개수
            output_size: 출력 데이터의 feature 개수
    """

    def __init__(self, input_size: int, output_size: int):
        super(CNN, self).__init__()
        
        # [a, b, c, ...] 형식의 input을 받아들이기 위해 Conv1d를 사용
        # conv1 층이 기대하는 input은 [배치 크기, 데이터 구분, 데이터 길이]이다.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        
        # conv2의 출력 크기를 계산
        conv2_output_size = self._get_conv_output(input_size)
        
        # 완전 연결 계층
        self.fc1 = nn.Linear(conv2_output_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def _get_conv_output(self, input_size):
        
        # 임의의 입력을 사용하여 conv2의 출력 크기를 계산
        x = torch.rand(1, 1, input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.numel()

    def forward(self, x):
        # unsqueeze를 통해 [batch_size, 2] -> [batch_size, 1, 2]로 변환
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))     # Conv layer 1 with ReLU
        x = F.relu(self.conv2(x))     # Conv layer 2 with ReLU
        
        x = x.view(x.size(0), -1)     # Flatten for fully connected layer
        
        x = F.relu(self.fc1(x))       # Fully connected layer 1 with ReLU
        x = self.fc2(x)               # Final output layer
        
        return x