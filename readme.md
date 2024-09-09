# IA_sub_template

## 개요

### 목적

1) 특정 Ion_max 내에서 User가 입력해주는 특정 Ion_in을 가능하게 하는 BSIM 모델 파라미터 조합
   이에 대한 답은 1개가 아니라 여러개가 나올것 같아요
2) Ion에 추가하여 Ioff (Ids @ Vgs=0V, Vds=Vdd)값을 동시에 targeting하는 1)번 수행

### 추가 목적

1) 템플릿을 하나 만들어서, 좀 쉽게 재활용할 수 있도록
2) optima같은 라이브러리를 도입해서 하이퍼파라미터 조정을 좀 더 쉽게 할 수 있도록

## 사용 방법

### Train_Data_Generator

안쪽에 있는 경로 설정 부분만 고치고 실행하면 됨!

## 주의 사항

git에 들어갈 코드들에는 model과 data, server_info를 포함하지 않고 있으니 그냥 실행하면 작동하지 않음!