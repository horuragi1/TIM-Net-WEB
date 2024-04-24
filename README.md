# MY_KOREAN_TIM-Net_SER(Speech Emotion Recognition)

![스크린샷, 2024-04-23 16-52-43](https://github.com/horuragi1/MY_KOREAN_TIM-Net/assets/102857746/a10e08d9-ad0e-41d4-8e21-996765ed0666)

아래 언급한 데이터셋 기준으로 validation accuracy에서 최고 성능은 89% 정도로 나왔고 10-fold validation 평균 성능은 86% 정도로 나왔습니다.

![스크린샷, 2024-04-24 13-30-28](https://github.com/horuragi1/MY_KOREAN_TIM-Net/assets/102857746/21157981-100d-4ff0-99a7-6655897bfbe4)


https://github.com/Jiaxin-Ye/TIM-Net_SER
위의 링크의 원본 TIM-Net을 ETRI KESDy18 데이터셋(https://nanum.etri.re.kr/share/kjnoh/SER-DB-ETRIv18?lang=ko_KR)과 감정 분류 데이터셋(https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=259)에서 추출한 음성 파일을 사용하여 학습시켰습니다.
또한 inference.py를 추가하여 모델 경로와 단일 음성 파일 경로를 입력하면 해당 모델이 예측한 라벨 별 확률 값을 출력할 수 있게 하였습니다.

# 1. 실행 방법

Code/inference.py 코드에서 wav_path는 감정을 인식하고 싶은 음성파일(.wav)의 경로로 변경하고, weight_path는 'Models/IEMOCAP_46_2024-04-23_15-37-31/10-fold_weights_best_1.hdf5' 또는 적절한 모델 파일(.hdf5)로 변경합니다.
python inference.py를 실행하면 2번째 사진과 같은 결과가 출력됩니다.
