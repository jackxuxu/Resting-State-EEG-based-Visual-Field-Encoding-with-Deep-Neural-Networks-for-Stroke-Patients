import scipy.io as sio
import h5py
import numpy as np
import keras
import math 
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from scipy.stats import entropy
from math import log2
def readEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131,500])
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            for index3 in range(len(file['alldata2'][index1][0][index2])):
                X[index1][index2][index3]=file['alldata2'][index1][0][index2][index3]
    return X
def readMeanEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T):
                sum+=file['alldata2'][index1][0][index2][index3]
            mean=sum/T
            X[index1][index2]=mean

    return X
def readPowerEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T):
                sum+=(file['alldata2'][index1][0][index2][index3])**2
            mean=sum/T
            X[index1][index2]=mean

    return X
def readStandardDeviationEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T):
                sum+=file['alldata2'][index1][0][index2][index3]
            mean=sum/T
            sum=0
            for index3 in range(T):
                sum+=(file['alldata2'][index1][0][index2][index3]-mean)**2
            X[index1][index2]=(sum/T)**(1/2)
    return X
def read1stDifferenceEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])

    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T-1):
                sum+=abs(file['alldata2'][index1][0][index2][index3+1]-file['alldata2'][index1][0][index2][index3])
            mean=sum/(T-1)
            X[index1][index2]=mean

    return X
def read2stDifferenceEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T-2):
                sum+=abs(file['alldata2'][index1][0][index2][index3+2]-file['alldata2'][index1][0][index2][index3])
            mean=sum/(T-2)
            X[index1][index2]=mean

    return X
def readNormalized1stDifferenceEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    S=readStandardDeviationEEG(file)
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T-1):
                sum+=abs(file['alldata2'][index1][0][index2][index3+1]-file['alldata2'][index1][0][index2][index3])
            mean=sum/(T-1)
            X[index1][index2]=mean
            X[index1][index2]=X[index1][index2]/S[index1][index2]
    return X
def readNormalized2stDifferenceEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    S=readStandardDeviationEEG(file)
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            sum=0
            mean=0
            T=len(file['alldata2'][index1][0][index2])
            for index3 in range(T-2):
                sum+=abs(file['alldata2'][index1][0][index2][index3+2]-file['alldata2'][index1][0][index2][index3])
            mean=sum/(T-2)
            X[index1][index2]=mean
            X[index1][index2]=X[index1][index2]/S[index1][index2]
    return X
def readEntropyEEG(file):
    X=np.zeros(shape=[len(file['alldata2']),131])
    for index1 in range(len(file['alldata2'])):
        for index2 in range(131):
            X[index1][index2]=file['alldata2'][index1][0][0][index2]

    return X
def readFrequencyEEG(file):
    X=np.zeros(shape=[len(file['alldata']),131,30,1])
    for index1 in range(len(file['alldata'])):
        for index2 in range(131):
            for index3 in range(30): 
                X[index1][index2][index3][0]=file['alldata'][index1][0][0][index2][index3]

    return X
def readFrequencyBandEEG(file):
    X=np.zeros(shape=[len(file['alldata']),128,6,1])
    #012 3456 789 1011  1220 2129
    band=[[0,3,3],[3,7,4],[7,10,3],[10,12,2],[12,21,9],[21,30,9]]
    for index1 in range(len(file['alldata'])):
        for index2 in range(128):
            for index3 in range(6): 
                sum=0
                for index4 in range(band[index3][0],band[index3][1]):
                    sum+=file['alldata'][index1][0][0][index2][index4]
                X[index1][index2][index3][0]=sum/band[index3][2]
    return X
def readConnectivityEEG(file):
    X=np.zeros(shape=[300,128,128])
    for index1 in range(300):
        for index2 in range(128):
            for index3 in range(128): 
                X[index1][index2][index3]=file['alldata'][0][index1][index2][index3]

    return X
#filecontrol=sio.loadmat('allcontrol_afterv6.mat')
#1-3 delta 4-7 ceta 8-10 lowalpha 10-12highalpha  13-21beta1 22-30beta2
# for index in range (24):
#     filename='allpre_afterv610connectivity'+str(index+1)+'.mat'
#     file=sio.loadmat(filename)
#     X=readConnectivityEEG(file)
#     filename='allpre_afterv610connectivity'+str(index+1)
#     np.save(filename, X) 
# print("finish pre")
# for index in range (24):
#     filename='allpost_afterv610connectivity'+str(index+1)+'.mat'
#     file=sio.loadmat(filename)
#     X=readConnectivityEEG(file)
#     filename='allpost_afterv610connectivity'+str(index+1)
#     np.save(filename, X) 
# print("finish post")    
# for index in range (24):
#     filename='allfu_afterv610connectivity'+str(index+1)+'.mat'
#     file=sio.loadmat(filename)
#     X=readConnectivityEEG(file)
#     filename='allfu_afterv610connectivity'+str(index+1)
#     np.save(filename, X) 
# print("finish fu")   
# for index in range (24):
#     filename='allcontrol_afterv610connectivity'+str(index+1)+'.mat'
#     file=sio.loadmat(filename)
#     X=readConnectivityEEG(file)
#     filename='allcontrol_afterv610connectivity'+str(index+1)
#     np.save(filename, X) 
# print("finish con")
for index in range (24):
    filename='allpre_afterv6freflip'+str(index+1)+'.mat'
    file=sio.loadmat(filename)
    X=readFrequencyBandEEG(file)
    filename='allpre_afterv6freflip6band'+str(index+1)
    np.save(filename, X) 
print("finish pre")
for index in range (24):
    filename='allpost_afterv6freflip'+str(index+1)+'.mat'
    file=sio.loadmat(filename)
    X=readFrequencyBandEEG(file)
    filename='allpost_afterv6freflip6band'+str(index+1)
    np.save(filename, X) 
print("finish post")    
for index in range (24):
    filename='allfu_afterv6freflip'+str(index+1)+'.mat'
    file=sio.loadmat(filename)
    X=readFrequencyBandEEG(file)
    filename='allfu_afterv6freflip6band'+str(index+1)
    np.save(filename, X) 
print("finish fu")   
for index in range (24):
    filename='allcontrol_afterv6freflip'+str(index+1)+'.mat'
    file=sio.loadmat(filename)
    X=readFrequencyBandEEG(file)
    filename='allcontrol_afterv6freflip6band'+str(index+1)
    np.save(filename, X) 
print("finish con")
# filepost=sio.loadmat('allpost_afterv6fre.mat')
# filepre=sio.loadmat('allpre_afterv6fre.mat')
# filefu=sio.loadmat('allfu_afterv6fre.mat')
# filecontrol=sio.loadmat('allcontrol_afterv6fre.mat')
# filepostflip=sio.loadmat('allpost_afterv6freflip.mat')
# filepreflip=sio.loadmat('allpre_afterv6freflip.mat')
# filefuflip=sio.loadmat('allfu_afterv6freflip.mat')
# filecontrolflip=sio.loadmat('allcontrol_afterv6freflip.mat')
print("start")
# X1=readFrequencyEEG(filepost)
# X2=readFrequencyEEG(filepre)
# X3=readFrequencyEEG(filefu)
# X4=readFrequencyEEG(filecontrol)
# X5=readFrequencyEEG(filepostflip)
# X6=readFrequencyEEG(filepreflip)
# X7=readFrequencyEEG(filefuflip)
# X8=readFrequencyEEG(filecontrolflip)
# print(X8.shape)
print("finish")
# np.save('allpost_afterv6fre', X1) 
# np.save('allpre_afterv6fre', X2)
# np.save('allfu_afterv6fre', X3) 
# np.save('allcontrol_afterv6fre', X4) 
# np.save('allpost_afterv6freflip', X5) 
# np.save('allpre_afterv6freflip', X6)
# np.save('allfu_afterv6freflip', X7) 
# np.save('allcontrol_afterv6freflip', X8) 
print("finish")





