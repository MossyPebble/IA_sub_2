import os, sys, json
import paramiko
import pandas as pd
from time import sleep

# lib 파일 인식을 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import SSHManager
from lib import data_preprocessor

# 실행 경로를 이 파일이 있는 경로로 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 연구실 서버 접속, shell 생성
server_info = json.load(open("../server_info.json", "r"))
ssh = SSHManager.SSHManager(server_info["host"], server_info["port"], server_info["id"], server_info["password"])
channel: paramiko.Channel = ssh.invoke_shell()
channel.settimeout(None)

############################################################### 경로 설정
# data를 위한 workspace 경로, 
data_server_workspace_path = '/home/mario/User/cws/'

# data를 뽑는데 사용할 .sp, .nmos 파일의 로컬 경로
high_sp_local_workspace_path = 'sp_and_nmos\\TFT_BSIM_high.sp'
low_sp_local_workspace_path = 'sp_and_nmos\\TFT_BSIM_low.sp'
nmos_local_workspace_path = 'sp_and_nmos\\BSIM_tft_cv.pmos'

local_paths: list = [high_sp_local_workspace_path, low_sp_local_workspace_path]

# data를 저장할 로컬 경로
data_local_workspace_path = 'data\\'
###############################################################


# 주어진 경로들을 순회하며, 각각의 .sp 파일을 서버로 전송하고, HSPICE를 실행하여 .ms0.csv 파일을 생성한다.
for path in local_paths:

    # 위 경로에서 파일 이름만 추출
    sp_local_name = path.split('\\')[-1]
    nmos_local_name = nmos_local_workspace_path.split('\\')[-1]

    sp_file_name = sp_local_name.split('.')[0]

    # .sp, .nmos 파일을 서버로 전송
    ssh.put_file(path, f'{data_server_workspace_path}{sp_local_name}')
    ssh.put_file(nmos_local_workspace_path, f'{data_server_workspace_path}{nmos_local_name}')

    sleep(1)

    ssh.execute_commands_over_shell(channel, [
        'cd /home/mario/User/cws/',
        f'hspice {sp_file_name}.sp'
    ])

    # .ms0.csv 파일을 로컬로 전송
    # 다만, 해당 파일이 생성되기까지 시간이 걸리므로, 1초 마다 파일이 생성되었는지 확인한다.
    while True:
        try:
            ssh.get_file(f'{data_server_workspace_path}{sp_file_name}.ms0.csv', f'{data_local_workspace_path}{sp_file_name}.csv')
            break
        except:
            sleep(1)

    # 서버에 생성된 .ms0.csv 파일은 삭제한다.
    ssh.execute_commands_over_shell(channel, [
        f'rm {data_server_workspace_path}{sp_file_name}.ms0.csv'
    ])

# 저장된 {data_local_workspace_path}data.csv를 불러오고 하나의 dataFrame으로 합친 다음 다시 csv로 저장한다.
datas = []
for path in local_paths:

    # 만약 local_paths의 길이가 1이라면 이 과정은 생략한다.
    if len(local_paths) == 1:
        break

    sp_file_name = path.split('\\')[-1].split('.')[0]
    data = data_preprocessor.transform_verilog_results_to_DataFrame(f'{data_local_workspace_path}{sp_file_name}.csv')
    datas.append(data)

data = pd.concat(datas, axis=1)
print(data.columns)

# data에 같은 이름의 컬럼이 있을 경우, 해당 컬럼 중 뒤에 있는 컬럼을 삭제한다.
data = data.loc[:,~data.columns.duplicated()]

data.to_csv(f'{data_local_workspace_path}data.csv')

# 서버와의 연결 종료
ssh.close()