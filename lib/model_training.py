import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys

def train_model(model: nn.Module, train_DataLoader: DataLoader, epochs: int, criterion: nn.Module, optimizer: torch.optim.Optimizer, test_DataLoader_for_accuracy: DataLoader=None, print_interval: int=1, early_stopping:bool=False) -> tuple[nn.Module, list, list]:
    
    """
        모델을 집어넣어 학습시키고 그 결과를 반환합니다.

        early_stopping 옵션은 test_DataLoader_for_accuracy가 주어졌을 때만 사용할 수 있습니다.

        Args:
            model (nn.Module): 학습할 모델
            train_DataLoader (DataLoader): 학습할 데이터
            epochs (int): 학습할 epoch 수
            criterion (nn.Module): 손실 함수
            optimizer (torch.optim.Optimizer): 최적화 함수
            test_DataLoader_for_accuracy (DataLoader): 테스트할 데이터
            print_interval (int): 출력할 epoch 간격
            early_stopping (bool): early stopping을 사용할지 여부, 이 옵션이 활성화되어 있다면, test loss가 증가하는 시점에서 학습을 중단한다.

        Returns:
            (nn.Module): 학습된 모델
            (list): 학습 데이터의 손실 함수 값
            (list): 테스트 데이터의 손실 함수 값
    """

    train_losses: list = []
    test_losses: list = []

    for epoch in range(epochs):
        
        for i, (x, y) in enumerate(train_DataLoader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        # 손실 함수 값 저장
        train_losses.append(loss.item())

        # 만약 테스트 데이터가 주어지지 않았다면, 테스트 데이터에 대한 loss와 accuracy를 계산하지 않는다.
        # 테스트 데이터가 주어졌다면, 테스트 데이터에 대한 loss와 accuracy를 계산해 저장한다.
        if test_DataLoader_for_accuracy == None:
            if epoch % print_interval == 0:
                print(f'Epoch {epoch}, Train Loss: {loss.item()}\n')
        else:
            test_loss = []
            test_accuracy = []
            with torch.no_grad():
                for i, (x, y) in enumerate(test_DataLoader_for_accuracy):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    test_loss.append(loss.item())
                    test_accuracy.append(torch.mean(torch.abs(y_pred - y) / y).item())
            test_losses.append(np.mean(test_loss))

            # 출력
            if epoch % print_interval == 0:
                print(f'Epoch {epoch}, Train Loss: {loss.item()}, \nTest Loss: {np.mean(test_loss)}, Test Accuracy: {np.mean(test_accuracy)}\n')

            # 만약 early stopping이 활성화되어 있다면, test loss가 증가하는 시점에서 학습을 중단한다.
            if early_stopping:

                # test loss가 증가하는 시점에서 학습을 중단한다.
                if len(test_losses) > 1 and test_losses[-1] > test_losses[-2]:
                    break

                # 또는 train loss가 증가하는 시점에서 학습을 중단한다.
                if len(train_losses) > 1 and train_losses[-1] > train_losses[-2]:
                    break

    return model, train_losses, test_losses