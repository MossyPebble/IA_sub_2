import os, sys, time, json
import paramiko

# lib 파일 인식을 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import SSHManager

# 실행 경로를 이 파일이 있는 경로로 변경
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 연구실 서버 접속
server_info = json.load(open("../server_info.json", "r"))
ssh = SSHManager.SSHManager(server_info["host"], server_info["port"], server_info["id"], server_info["password"])


############################################################### 경로 설정
# data를 위한 workspace 경로, 
data_server_workspace_path = '/home/mario/User/cws/'

# data를 뽑는데 사용할 .sp, .nmos 파일의 로컬 경로
sp_local_workspace_path = 'sp_and_nmos\\TFT_BSIM_low.sp'
nmos_local_workspace_path = 'sp_and_nmos\\BSIM_tft_cv.pmos'

# data를 저장할 로컬 경로
data_local_workspace_path = 'data\\'
###############################################################


# 위 경로에서 파일 이름만 추출
sp_local_name = sp_local_workspace_path.split('\\')[-1]
nmos_local_name = nmos_local_workspace_path.split('\\')[-1]

sp_file_name = sp_local_name.split('.')[0]

# .sp, .nmos 파일을 서버로 전송
ssh.put_file(sp_local_workspace_path, f'{data_server_workspace_path}{sp_local_name}')
ssh.put_file(nmos_local_workspace_path, f'{data_server_workspace_path}{nmos_local_name}')

# HSPICE를 실행하여 .ms0.csv 파일을 생성
channel: paramiko.Channel = ssh.invoke_shell()
channel.settimeout(None)

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
        time.sleep(1)

# 서버와의 연결 종료
ssh.close()