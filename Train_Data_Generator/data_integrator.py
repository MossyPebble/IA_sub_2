import os, sys
import pandas as pd

# lib 파일 인식을 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import SSHManager
from lib import data_preprocessor

# 실행 경로를 이 파일이 있는 경로로 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# data를 뽑는데 사용할 .sp, .nmos 파일의 로컬 경로
high_sp_local_workspace_path = 'sp_and_nmos/TFT_BSIM_high.sp'
low_sp_local_workspace_path = 'sp_and_nmos/TFT_BSIM_low.sp'
nmos_local_workspace_path = 'sp_and_nmos/BSIM_tft_cv.pmos'
data_local_workspace_path = 'data/'

local_paths: list = [high_sp_local_workspace_path, low_sp_local_workspace_path]

# 저장된 {data_local_workspace_path}data.csv를 불러오고 하나의 dataFrame으로 합친 다음 다시 csv로 저장한다.
datas = []
for path in local_paths:

    # 만약 local_paths의 길이가 1이라면 이 과정은 생략한다.
    if len(local_paths) == 1:
        break

    sp_file_name = path.split('/')[-1].split('.')[0]
    data = data_preprocessor.transform_verilog_results_to_DataFrame(f'{data_local_workspace_path}{sp_file_name}.csv')
    datas.append(data)

data = pd.concat(datas, axis=1)
print(data.columns)

# data에 같은 이름의 컬럼이 있을 경우, 해당 컬럼 중 뒤에 있는 컬럼을 삭제한다.
data = data.loc[:,~data.columns.duplicated()]

data.to_csv(f'{data_local_workspace_path}data.csv')