# MY_KOREAN_TIM-Net

![스크린샷, 2024-04-24 13-30-28](https://github.com/horuragi1/MY_KOREAN_TIM-Net/assets/102857746/21157981-100d-4ff0-99a7-6655897bfbe4)


https://github.com/Jiaxin-Ye/TIM-Net_SER
위의 링크의 원본 TIM-Net을 ETRI KESDy18 데이터셋(https://nanum.etri.re.kr/share/kjnoh/SER-DB-ETRIv18?lang=ko_KR)과 감정 분류 데이터셋(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=259)에서 추출한 음성 파일을 사용하여 학습시켰습니다.
또한 inference.py를 추가하여 모델 경로와 단일 음성 파일 경로를 입력하면 해당 모델이 예측한 라벨 별 확률 값을 출력할 수 있게 하였습니다.
