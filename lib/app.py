from lib import ANN, SSHManager
import numpy as np
import pandas as pd
import os, sys

def get_value_with_parameter(
        paramter: dict[str, float], 
        sshmanager: SSHManager, 
        server_file_path: str, 
        sp_file_path: str,
        mos_file_path: str,
        parameter_names: list[str]
        ) -> None:

    """
        dict 형태로 파라미터를 전달 받아, 연구실 서버에 전달해 HSPICE를 실행하고 파일을 받는 함수

        Args:
            paramter (dict[str, float]): HSPICE에 전달할 파라미터
            sshmanager (SSHManager.SSHManager): 연구실 서버에 접속하기 위한 SSHManager 객체
            server_path (str): 서버 측 파일 경로
            sp_file_path (list[str]): sp 파일 경로
            mos_file_path (str): mos 파일 경로
            parameter_names (list[str]): 파라미터 이름 리스트
    """

    sp_file_name = sp_file_path.split('/')[-1]
    mos_file_name = mos_file_path.split('/')[-1]

    # 바로 위에서 복사한 파일의 내용을 변경한다.
    file_content = []
    with open(mos_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        
        for name in parameter_names:
            if f'r_{name}' in line:
                line = f' + {name.upper()} = {paramter[name]}\n'
                break

        # line 안에서 해당 부분만을 찾아 값을 변경한다.
        # if 'r_phig' in line:
        #     line = f' + PHIG = {paramter["phig"]}\n'
        # if 'r_cit' in line:
        #     line = f' + CIT = {paramter["cit"]}\n'
        # if 'r_u0' in line:
        #     line = f' + U0 = {paramter["u0"]}\n'
        # if 'r_ua' in line:
        #     line = f' + UA = {paramter["ua"]}\n'
        # if 'r_eu' in line:
        #     line = f' + EU = {paramter["eu"]}\n'
        # if 'r_etamob' in line:
        #     line = f' + ETAMOB = {paramter["etamob"]}\n'
        # if 'r_up' in line:
        #     line = f' + UP = {paramter["up"]}\n'
        # if 'r_rdsw' in line:
        #     line = f' + RDSW = {paramter["rdsw"]}\n'

        file_content.append(line)

    # 변경된 내용으로 새로운 파일을 생성한다.
    with open(mos_file_name, 'w') as f:
        f.writelines(file_content)

    # 서버에 파일을 전송한다.
    sshmanager.put_file(sp_file_path, f"{server_file_path}{sp_file_name}")
    sshmanager.put_file(mos_file_name, f"{server_file_path}{mos_file_name}")

    # HSPICE를 실행한다.
    result = sshmanager.send_command(f"hspice -i {server_file_path}{sp_file_name} > {server_file_path}result.txt")

    # 서버에서 결과 파일을 로컬로 전송한다.
    sshmanager.get_file(f"{server_file_path}/result.txt", "result.txt")

    # 기존 파일을 삭제한다.
    sshmanager.send_command(f"rm {server_file_path}{sp_file_name}")
    sshmanager.send_command(f"rm {server_file_path}{mos_file_name}")
    sshmanager.send_command(f"rm {server_file_path}result.txt")
    os.remove(mos_file_name)


def get_value_with_model(models: list[ANN.ANN], input: pd.Series) -> list[np.ndarray]:

    """
        ANN 모델과 파라미터를 전달 받아, ANN 모델을 통해 결과를 반환하는 함수

        한 번에 여러 ANN 모델을 사용할 것이기 때문에 ANN 모델을 list 형태로 전달 받는다.

        Args:
            model (list[ANN.ANN]): ANN 모델
            input (pd.Series): ANN 모델에 전달할 입력값, DataLoader를 통해 얻어오는 탓에 pd.Series 형태로 전달 받는다.

        Returns:
            list[np.ndarray]: ANN 모델을 통해 반환된 결과값
    """

    # ANN 모델을 통해 결과를 반환한다.
    result = []
    for model in models:
        pred = model.predict(input)

        # pred를 numpy 배열로 변환하여 result에 추가한다.
        pred_numpy = pred.detach().numpy().squeeze()
        result.append(pred_numpy)

    return result

def remove_unit(file_content: list[str], search_key: list[str]=['i' + str(i) for i in range(63)]) -> pd.DataFrame:

    """
        문자열에서 단위를 제거하는 함수

        입력으로 받는 문자열은 i0~i62까지의 전류값을 가지고 있고, 해당 값들을 순서대로 pd.DataFrame으로 반환한다.

        Args:
            file_content (list[str]): 전류값을 가지고 있는 문자열
            search_key (list[str]): 값을 찾기 위한 키워드

        Returns:
            pd.DataFrame: 전류값을 가지고 있는 pd.DataFrame
    """

    result = {}
    index = 0

    for line in file_content:
        for key in search_key:
            if key in line and key not in result.keys():
                current = line.split('=')[-1]
                if 'm' in current:
                    current = current.replace('m', 'e-3')
                elif 'u' in current:
                    current = current.replace('u', 'e-6')
                elif 'n' in current:
                    current = current.replace('n', 'e-9')
                elif 'p' in current:
                    current = current.replace('p', 'e-12')
                elif 'f' in current:
                    current = current.replace('f', 'e-15')
                elif 'a' in current:
                    current = current.replace('a', 'e-18')

                # 현재 line에서 전류값을 찾았으므로, 해당 값을 result에 추가한다.
                result[key] = float(current)
                index += 1

                # 다음 line으로 넘어간다.
                break
    
    # result의 key를 column으로, value를 row로 하는 pd.DataFrame을 생성한다.
    result = pd.DataFrame(result, index=[0])

    return result