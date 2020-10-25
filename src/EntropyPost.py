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
#filecontrol=sio.loadmat('allcontrol_afterv6.mat')
#1-3 delta 4-7 ceta 8-10 lowalpha 10-12highalpha  13-21beta1 22-30beta2
filepost1=sio.loadmat('allpost_afterv6entropy1-3.mat')
filepost2=sio.loadmat('allpost_afterv6entropy4-7.mat')
filepost3=sio.loadmat('allpost_afterv6entropy8-10.mat')
filepost4=sio.loadmat('allpost_afterv6entropy10-12.mat')
filepost5=sio.loadmat('allpost_afterv6entropy13-21.mat')
filepost6=sio.loadmat('allpost_afterv6entropy22-30.mat')
filepre1=sio.loadmat('allpre_afterv6entropy1-3.mat')
filepre2=sio.loadmat('allpre_afterv6entropy4-7.mat')
filepre3=sio.loadmat('allpre_afterv6entropy8-10.mat')
filepre4=sio.loadmat('allpre_afterv6entropy10-12.mat')
filepre5=sio.loadmat('allpre_afterv6entropy13-21.mat')
filepre6=sio.loadmat('allpre_afterv6entropy22-30.mat')
filefu1=sio.loadmat('allfu_afterv6entropy1-3.mat')
filefu2=sio.loadmat('allfu_afterv6entropy4-7.mat')
filefu3=sio.loadmat('allfu_afterv6entropy8-10.mat')
filefu4=sio.loadmat('allfu_afterv6entropy10-12.mat')
filefu5=sio.loadmat('allfu_afterv6entropy13-21.mat')
filefu6=sio.loadmat('allfu_afterv6entropy22-30.mat')
filepre=sio.loadmat('allpre_afterv6entropy.mat')
filepost=sio.loadmat('allpost_afterv6entropy.mat')
filefu=sio.loadmat('allfu_afterv6entropy.mat')
print("start")
X1=readEntropyEEG(filepost1)
X2=readEntropyEEG(filepost2)
X3=readEntropyEEG(filepost3)
X4=readEntropyEEG(filepost4)
X5=readEntropyEEG(filepost5)
X6=readEntropyEEG(filepost6)
X7=readEntropyEEG(filepre1)
X8=readEntropyEEG(filepre2)
X9=readEntropyEEG(filepre3)
X10=readEntropyEEG(filepre4)
X11=readEntropyEEG(filepre5)
X12=readEntropyEEG(filepre6)
X13=readEntropyEEG(filefu1)
X14=readEntropyEEG(filefu2)
X15=readEntropyEEG(filefu3)
X16=readEntropyEEG(filefu4)
X17=readEntropyEEG(filefu5)
X18=readEntropyEEG(filefu6)
X16=readEntropyEEG(filefu4)
X17=readEntropyEEG(filefu5)
X18=readEntropyEEG(filefu6)
X19=readEntropyEEG(filepre)
X20=readEntropyEEG(filepost)
X21=readEntropyEEG(filefu)
print("finish")
#np.savetxt('allpost_afterv6entropy.csv', X1, delimiter=',')
np.save('allpost_afterv6entropy1-3', X1) 
np.save('allpost_afterv6entropy4-7', X2) 
np.save('allpost_afterv6entropy8-10', X3) 
np.save('allpost_afterv6entropy10-12', X4) 
np.save('allpost_afterv6entropy13-21', X5) 
np.save('allpost_afterv6entropy22-30', X6) 
#np.savetxt('alpre_afterv6entropy.csv', X2, delimiter=',')
np.save('allpre_afterv6entropy1-3', X7)
np.save('allpre_afterv6entropy4-7', X8)
np.save('allpre_afterv6entropy8-10', X9)
np.save('allpre_afterv6entropy10-12', X10)
np.save('allpre_afterv6entropy13-21', X11)
np.save('allpre_afterv6entropy22-30', X12)
#np.savetxt('allfu_afterv6entropy.csv', X3, delimiter=',')
np.save('allfu_afterv6entropy1-3', X13) 
np.save('allfu_afterv6entropy4-7', X14)
np.save('allfu_afterv6entropy8-10', X15)
np.save('allfu_afterv6entropy10-12', X16)
np.save('allfu_afterv6entropy13-21', X17)
np.save('allfu_afterv6entropy22-30', X18)
np.save('allpre_afterv6entropy', X19)
np.save('allpost_afterv6entropy', X20)
np.save('allfu_afterv6entropy', X21)

print("finish")





