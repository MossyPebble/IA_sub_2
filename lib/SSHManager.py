import paramiko, time

class SSHManager:
    
    """
    어떤 ssh 서버에 접속하고 그 안에서 명령어 실행, 파일 송수신을 담당하는 클래스
    """

    def __init__(self, host, port, userId, password) -> None:

        """
        ssh 서버에 접속한다.

        Args:
            host (str): 접속할 ssh 서버의 주소
            port (int): 접속할 ssh 서버의 포트
            userId (str): ssh 서버에 접속할 계정
            password (str): ssh 서버에 접속할 계정의 비밀번호
        """
    
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(host, port, userId, password)
            self.sftp = self.ssh.open_sftp()
        except Exception as e:
            print(e)
            raise Exception("ssh 서버에 접속할 수 없습니다. 인터넷 연결 상태를 확인해주세요.")
        
    def invoke_shell(self) -> paramiko.Channel:

        """
        ssh 서버에 shell을 실행한다.

        만약, 명령어 실행 경로를 계속 유지하고 싶다거나 하면 shell을 열어서 명령어를 실행하면 된다.

        Returns:
            paramiko.Channel: shell을 실행한 결과
        """

        return self.ssh.invoke_shell()
    
    def execute_commands_over_shell(self, channel: paramiko.Channel, commands: list, debug: bool=False) -> None:

        """
        SSH 채널을 통해 명령어를 실행하는 함수

        Args:
            channel (paramiko.Channel): SSH 채널

            commands (list): 실행할 명령어 리스트

            debug (bool, optional): 디버깅 여부. Defaults to False.
        """

        for command in commands:
            if debug: print(f"Sending command: {command.strip()}")
            channel.send(command + '\n') 
            
            while True:
                if channel.recv_ready():
                    output = ""
                    output += channel.recv(2^20).decode('utf-8')
                    print(output, end="")
                else:
                    time.sleep(2) # 각 명령어 후 충분한 대기 시간 추가
                    if not channel.recv_ready():
                        break

    def get_file(self, src, dst) -> None:

        """
        ssh 서버로부터 파일을 다운로드한다.

        Args:
            src (str): 다운로드할 파일의 경로 (서버)

            dst (str): 다운로드한 파일을 저장할 경로 (로컬)
        """

        self.sftp.get(src, dst)

    def put_file(self, src: str, dst: str) -> None:

        """
        ssh 서버로 파일을 업로드한다.

        Args:
            src (str): 업로드할 파일의 경로 (로컬)

            dst (str): 업로드할 파일을 저장할 경로 (서버)
        """

        self.sftp.put(src, dst)

    def send_command(self, cmd: str, debug: bool=False) -> str:

        """
        ssh 서버에 명령어를 전송한다.

        Args:
            cmd (str): 전송할 명령어

        Returns:
            str: 명령어의 실행 결과

            debug (bool, optional): 디버깅 여부. Defaults to False.
        """

        if debug: print("명령어:", cmd)
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        return stdout.read().decode()
    
    def change_file_content(self, file_path, old, new) -> str:

        """
        ssh 서버에 있는 파일의 내용을 변경한다.

        Args:
            file_path (str): 내용을 변경할 파일의 경로

            old (str): 변경할 내용 중 변경 전 내용
            
            new (str): 변경할 내용 중 변경 후 내용

        Returns:
            str: 명령어의 실행 결과
        """

        return self.send_command(f"sed -i \"s/{old}/{new}/g\" {file_path}")

    def close(self) -> None:
        self.sftp.close()
        self.ssh.close()

    def __del__(self) -> None:
        self.sftp.close()
        self.ssh.close()