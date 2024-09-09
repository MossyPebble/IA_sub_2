import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

def transform_verilog_results_to_DataFrame(file_path: str, skiprows: int=4) -> pd.DataFrame:

    """
        베릴로그 시뮬레이션 결과로 얻은 csv 파일을 읽어서 DataFrame으로 변환하는 함수

        첫 번째 행인 index, 마지막 2행 temper, alter#은 불필요한 정보이므로 제거한다.

        이 후, DataFrame 안의 값들을 str에서 float으로 변환한다.

        Args:
            file_path (str): 베릴로그 시뮬레이션 결과로 얻은 csv 파일의 경로

        Returns:
            (pd.DataFrame): 베릴로그 시뮬레이션 결과를 DataFrame으로 변환한 결과
    """

    df = pd.read_csv(file_path, header=0, skiprows=skiprows)
    if 'index' in df.columns and 'temper' in df.columns and 'alter#' in df.columns:
        df = df.drop(['index', 'temper', 'alter#'], axis=1)
    df = df.apply(pd.to_numeric)
    return df

def normalize_DataFrame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    """
        DataFrame의 데이터를 정규화하는 함수

        오류값이 있다면 먼저 해당 열을 제거한다.

        추 후 역정규화를 위해 최대값과 최소값 또한 반환한다.

        Args:
            df (pd.DataFrame): 정규화할 DataFrame

        Returns:
            (pd.DataFrame): 정규화된 DataFrame

            (pd.Series): 정규화된 DataFrame의 최소값

            (pd.Series): 정규화된 DataFrame의 최대값
    """

    min = df.min()
    max = df.max()
    df = (df - min) / (max - min)

    # NaN이 있는 열은 제거하고, 몇 번째 열이 제거되었는지 저장한다.
    droped_columns = df.columns[df.isna().any()].tolist()
    df = df.dropna(axis=1)
    print(f'{droped_columns} columns are dropped.')

    # 제거된 열은 최대값과 최소값에서도 제거한다.
    min = min.drop(droped_columns)
    max = max.drop(droped_columns)

    # 최대값과 최소값도 dataframe으로 변환한다.
    min = pd.Series(min.values, index=df.columns)
    max = pd.Series(max.values, index=df.columns)

    return df, min, max

def transform_DataFrame_to_DataLoader(input_df: pd.DataFrame, output_df: pd.DataFrame, batch_size: int, shuffle: bool=True) -> DataLoader:

    """
        DataFrame을 DataLoader로 변환하는 함수

        Args:
            input_df (pd.DataFrame): DataLoader에 넣을 input 데이터
            output_df (pd.DataFrame): DataLoader에 넣을 output 데이터
            batch_size (int): DataLoader에 넣을 배치 크기
            shuffle (bool): DataLoader에 넣을 데이터를 섞을지 여부
            input_output_border (int): input과 output의 경계를 나타내는 인덱스

        Returns:
            (DataLoader): 변환된 DataLoader
    """

    class CustomDataset(Dataset):
        def __init__(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
            self.input = input_df
            self.output = output_df

        def __len__(self):
            return len(self.input)

        def __getitem__(self, idx):
            x = torch.tensor(self.input.iloc[idx, :].values, dtype=torch.float32)
            y = torch.tensor(self.output.iloc[idx, :].values, dtype=torch.float32)
            return x, y
        
    dataset = CustomDataset(input_df, output_df)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)