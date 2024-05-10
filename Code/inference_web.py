'''
wav_path : 음성파일 경로(.wav)
weight_path : 모델 경로(.hdf5)

위의 두가지만 각자 경로에 맞게 수정하시면 됩니다.

모델 파일은 'Code/Models/IEMOCAP_46_2024-04-23_15-37-31'에 있는 10가지 .hdf5 파일 중 아무거나 하나 사용하시면 됩니다.
'''

import librosa
import numpy as np
import tensorflow.keras.backend as K
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd
import copy
import pyaudio
import numpy as np
import wave
import streamlit as st
from st_audiorec import st_audiorec
from audio_recorder_streamlit import audio_recorder

from TIMNET import TIMNET
from Model import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
parser.add_argument('--test_path', type=str, default='./Test_Models/EMODB_46')
parser.add_argument('--data', type=str, default='IEMOCAP')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10
    
def get_feature(file_path: str, feature_type:str="MFCC", mean_signal_length:int=96000, embed_len: int = 39):
    feature = None
    signal, fs = librosa.load(file_path)# Default setting on sampling rate
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    if feature_type == "MFCC":
        mfcc =  librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=embed_len)
        feature = np.transpose(mfcc)
    return feature
    
input_shape = (606, 39)

IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap

CLASS_LABELS_dict = {
               "IEMOCAP": IEMOCAP_CLASS_LABELS
}

CLASS_LABELS = CLASS_LABELS_dict[args.data]

#@st.experimental_memo

cnt = 0

def load_model_once(_args, input_shape, class_labels):
    MyModel = TIMNET_Model(args=args, input_shape=input_shape, class_label=class_labels)
    MyModel.create_model()
    weight_path = 'Code/Models/IEMOCAP_46_2024-04-23_15-37-31/10-fold_weights_best_1.hdf5'
    MyModel.model.load_weights(weight_path)
    return MyModel




# TIMNET 모델 로드 (한 번만 실행)

if cnt == 0:
    MyModel = load_model_once(_args=args, input_shape=input_shape, class_labels=CLASS_LABELS)
    cnt = 1

# 나머지 애플리케이션 코드 작성
# ...
title = "TIM-Net 음성 감정 인식"
st.title(title)

genre = st.radio(
    "입력 모드 선택",
    [":rainbow[마이크]", "***파일 업로드***"],
    captions = ["마이크를 사용하여 음성을 녹음합니다.", "wav 파일을 업로드하여 감정을 분석합니다."])

if genre == ":rainbow[마이크]":
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000, energy_threshold = 0)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # 파일 저장
        output_path = "output.wav"
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        st.success(f"Audio saved to {output_path}")
        st.success("녹음 완료!")
            
        if os.path.getsize(output_path):
            
            with st.spinner('감정 인식 중...'):
                wav_path = 'output.wav'

                return_feature = get_feature(file_path=wav_path, mean_signal_length=310000).reshape((-1, 606, 39))
                prediction = MyModel.model.predict(return_feature)
                #print('angry     happy     neutral   sad       ')
                #print(prediction)

            st.write('angry happy neutral sad')
            st.write(prediction)

            #st.audio(wav_path, format='audio/wav')
else:
    
    wav_file = st.file_uploader('음성 파일(.wav)을 업로드 하세요.', type=['wav'])
    
    if wav_file != None:
        wav_file.name = 'output.wav'
    
        with open(wav_file.name, 'wb') as f: #해당 경로의 폴더에서 파일의 이름으로 생성하겠다.
            f.write(wav_file.getbuffer()) # 해당 내용은 Buffer로 작성하겠다.
                # 기본적으로 이미즈는 buffer로 저장되고 출력할때도 buffer로 출력한다.
                
        output_path = wav_file.name
            
        if os.path.getsize(output_path):
            
            with st.spinner('감정 인식 중...'):
                wav_path = 'output.wav'

                return_feature = get_feature(file_path=wav_path, mean_signal_length=310000).reshape((-1, 606, 39))
                prediction = MyModel.model.predict(return_feature)
                #print('angry     happy     neutral   sad       ')
                #print(prediction)
                
            st.audio(wav_path, format='audio/wav')

            st.write('angry happy neutral sad')
            st.write(prediction)

            #st.audio(wav_path, format='audio/wav')



    


    
    
    




























