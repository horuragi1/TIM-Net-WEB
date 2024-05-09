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

MyModel = TIMNET_Model(args=args, input_shape=input_shape, class_label=CLASS_LABELS)

MyModel.create_model()
weight_path = 'Code/Models/IEMOCAP_46_2024-04-23_15-37-31/10-fold_weights_best_1.hdf5'
MyModel.model.load_weights(weight_path)












title = "TIM-Net 음성 감정 인식"
st.title(title)
if st.button('녹음'):
    with st.spinner('5초간 녹음 중...'):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "outputtt.wav"

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    st.success("녹음 완료!")
        
    with st.spinner('감정 인식 중...'):
        wav_path = 'outputtt.wav'

        return_feature = get_feature(file_path=wav_path, mean_signal_length=310000).reshape((-1, 606, 39))
        prediction = MyModel.model.predict(return_feature)
        #print('angry     happy     neutral   sad       ')
        #print(prediction)
    
    st.write('angry happy neutral sad')
    st.write(prediction)
    
    st.audio(wav_path, format='audio/wav')
    
    


    
    
    




























