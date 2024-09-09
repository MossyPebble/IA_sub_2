import os, sys
import pandas as pd, numpy as np
from lib.SSHManager import SSHManager
from lib import app

def get_test_data_from_server(ssh: SSHManager, sp_file_paths: list[str], mos_file_path: str, parameter_names: list[str], parameters:np.ndarray, file_split_line: int=1200) -> pd.DataFrame:

    """
    두개의 sp 파일을 활용해서 데이터를 가져와야 하는 경우가 생겨서 한 함수 안에서 처리하도록 만듦.

    Args:
        ssh (SSHManager): ssh 객체
        sp_file_paths (list[str]): sp 파일의 경로 리스트
        mos_file_path (str): mos 파일의 경로
        parameters (np.ndarray): 파라미터 리스트, 해당 파라미터들은 unnormailzed 되어있어야 함.
    """

    outputs = []

    for sp_file_path in sp_file_paths:
        app.get_value_with_parameter(parameters, ssh, "/home/mario/User/cws/test/", sp_file_path, mos_file_path, parameter_names)
        with open("result.txt", "r") as f:
            result = f.readlines()

        # HSPICE 실행 결과에서 i값을 추출한다.
        result = result[file_split_line:]
        output = app.remove_unit(result, parameter_names)
        outputs.append(output)

        # 결과 파일을 삭제한다.
        os.remove("result.txt")
        ssh.send_command("rm /home/mario/User/cws/test/result.txt")
        
    # output에서 중복되는 column을 제거한다.
    output = pd.concat(outputs, axis=1)
    output = output.loc[:,~output.columns.duplicated()]

    return output