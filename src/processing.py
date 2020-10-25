import scipy.io as sio
import h5py
import numpy as np
import keras

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
#filecontrol=sio.loadmat('allcontrol_afterv6.mat')
filecontrol=sio.loadmat('allpre_afterv6.mat')
print(filecontrol['alldata2'][1][0].shape)
# filefu=sio.loadmat('allfu_afterv6.mat')
# 
# filepost=sio.loadmat('allpost_afterv6.mat')
# 
# filepre=sio.loadmat('allpre_afterv6.mat')
print("start")
X1=readPowerEEG(filecontrol)
print("finish1")
# X2=readMeanEEG(filecontrol)
# print("finish2")
# X3=readStandardDeviationEEG(filecontrol)
# print("finish3")
# X4=read1stDifferenceEEG(filecontrol)
# print("finish4")
# X5=readNormalized1stDifferenceEEG(filecontrol)
# print("finish5")
# X6=read2stDifferenceEEG(filecontrol)
# print("finish6")
# X7=readNormalized2stDifferenceEEG(filecontrol)
# print("finish7")
# 
# np.savetxt('allpre_afterv6Power.csv', X1, delimiter=',')
# np.save('allpre_afterv6Power', X1) 
# np.savetxt('allpre_afterv6Mean.csv', X2, delimiter=',')
# np.save('allpre_afterv6Mean', X2)
# np.savetxt('allpre_afterv6StandardDeviation.csv', X3, delimiter=',')
# np.save('allpre_afterv6StandardDeviation', X3) 
# np.savetxt('allpre_afterv61stDifference.csv', X4, delimiter=',')
# np.save('allpre_afterv61stDifference', X4) 
# np.savetxt('allpre_afterv6Normalized1stDifference.csv', X5, delimiter=',')
# np.save('allpre_afterv6Normalized1stDifference', X5) 
# np.savetxt('allpre_afterv62stDifference.csv', X6, delimiter=',')
# np.save('allpre_afterv62stDifference', X6) 
# np.savetxt('allpre_afterv6Normalized2stDifference.csv', X7, delimiter=',')
# np.save('allpre_afterv6Normalized2stDifference', X7) 
 
print(filecontrol['alldata2'].shape)

print("finish")

fileresp=sio.loadmat('allresp.mat')
percentb=np.zeros(shape=[24,1])
percentg=np.zeros(shape=[24,1])
percentw=np.zeros(shape=[24,1])
for index in range(0,24):
    percentb[index]=fileresp['allresp'][0][index][0][0]/441
    percentg[index]=fileresp['allresp'][0][index][0][1]/441
    percentw[index]=fileresp['allresp'][0][index][0][2]/441
print(percentb)
print(percentg)
print(percentw)
percentControlb=np.zeros(shape=[len(X1)])
percentControlg=np.zeros(shape=[len(X1)])
percentControlw=np.zeros(shape=[len(X1)])
percentControlbgw=np.zeros(shape=[len(X1),3])
for index in range(len(X1)):
    number=filecontrol['alldata2'][index][1]-1
    percentControlb[index]=percentb[number]*100
    percentControlg[index]=percentg[number]*100
    percentControlw[index]=percentw[number]*100
    percentControlbgw[index][0]=percentb[number]*100
    percentControlbgw[index][1]=percentg[number]*100
    percentControlbgw[index][2]=percentw[number]*100
print(percentControlb.shape)
print(percentControlg.shape)
print(percentControlw.shape)
print(percentControlbgw.shape)
np.savetxt('allpre_afterv6respB.csv', percentControlb, delimiter=',')
np.save('allpre_afterv6respB', percentControlb) 
np.savetxt('allpre_afterv6respG.csv', percentControlg, delimiter=',')
np.save('allpre_afterv6respG', percentControlg) 
np.savetxt('allpre_afterv6respW.csv', percentControlw, delimiter=',')
np.save('allpre_afterv6respW', percentControlw) 
np.savetxt('allpre_afterv6resp.csv', percentControlbgw, delimiter=',')
np.save('allpre_afterv6resp', percentControlbgw) 













