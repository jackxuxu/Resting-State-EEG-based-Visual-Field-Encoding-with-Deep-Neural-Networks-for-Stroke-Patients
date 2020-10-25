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
#1-3 delta 4-7 ceta 8-10 lowalpha 10-12highalpha  13-21beta1 22-30beta2
# filepredelta=sio.loadmat('allpre_afterv6flip1-3.mat')
# filepreceta=sio.loadmat('allpre_afterv6flip4-7.mat')
# fileprelowalpha=sio.loadmat('allpre_afterv6flip8-10.mat')
# fileprehighalpha=sio.loadmat('allpre_afterv6flip10-12.mat')
# fileprebeta1=sio.loadmat('allpre_afterv6flip13-21.mat')
# fileprebeta2=sio.loadmat('allpre_afterv6flip22-30.mat')
  
# filepostdelta=sio.loadmat('allpost_afterv6flip1-3.mat')
# filepostceta=sio.loadmat('allpost_afterv6flip4-7.mat')
# filepostlowalpha=sio.loadmat('allpost_afterv6flip8-10.mat')
# fileposthighalpha=sio.loadmat('allpost_afterv6flip10-12.mat')
# filepostbeta1=sio.loadmat('allpost_afterv6flip13-21.mat')
# filepostbeta2=sio.loadmat('allpost_afterv6flip22-30.mat')
# 
filefudelta=sio.loadmat('allfu_afterv6flip1-3.mat')
filefuceta=sio.loadmat('allfu_afterv6flip4-7.mat')
filefulowalpha=sio.loadmat('allfu_afterv6flip8-10.mat')
filefuhighalpha=sio.loadmat('allfu_afterv6flip10-12.mat')
filefubeta1=sio.loadmat('allfu_afterv6flip13-21.mat')
filefubeta2=sio.loadmat('allfu_afterv6flip22-30.mat')
filepre=sio.loadmat('allpre_afterv6flip.mat')
filepost=sio.loadmat('allpost_afterv6flip.mat')
filefu=sio.loadmat('allfu_afterv6flip.mat')

# filepredelta=sio.loadmat('allpre_after1-3v6.mat')
# filepreceta=sio.loadmat('allpre_after4-7v6.mat')
# fileprelowalpha=sio.loadmat('allpre_after8-10v6.mat')
# fileprehighalpha=sio.loadmat('allpre_after10-12v6.mat')
# fileprebeta1=sio.loadmat('allpre_after13-21v6.mat')
# fileprebeta2=sio.loadmat('allpre_after22-30v6.mat')
#  
# filepostdelta=sio.loadmat('allpost_after1-3v6.mat')
# filepostceta=sio.loadmat('allpost_after4-7v6.mat')
# filepostlowalpha=sio.loadmat('allpost_after8-10v6.mat')
# fileposthighalpha=sio.loadmat('allpost_after10-12v6.mat')
# filepostbeta1=sio.loadmat('allpost_after13-21v6.mat')
# filepostbeta2=sio.loadmat('allpost_after22-30v6.mat')
# 
# filefudelta=sio.loadmat('allfu_after1-3v6.mat')
# filefuceta=sio.loadmat('allfu_after4-7v6.mat')
# filefulowalpha=sio.loadmat('allfu_after8-10v6.mat')
# filefuhighalpha=sio.loadmat('allfu_after10-12v6.mat')
# filefubeta1=sio.loadmat('allfu_after13-21v6.mat')
# filefubeta2=sio.loadmat('allfu_after22-30v6.mat')
# filepost=sio.loadmat('allpost_afterv6.mat')
# filefu=sio.loadmat('allfu_afterv6.mat')
# filecontrol=sio.loadmat('allcontrol_afterv6.mat')
#print(filecontrol['alldata2'][1][0].shape)
# filefu=sio.loadmat('allfu_afterv6.mat')
# 
# filepost=sio.loadmat('allpost_afterv6.mat')
# 
# filepre=sio.loadmat('allpre_afterv6.mat')
print("start")
# X1=readNormalized1stDifferenceEEG(filepredelta)
# print("finish1")
# X2=readNormalized1stDifferenceEEG(filepreceta)
# print("finish2")
# X3=readNormalized1stDifferenceEEG(fileprelowalpha)
# print("finish3")
# X4=readNormalized1stDifferenceEEG(fileprehighalpha)
# print("finish4")
# X5=readNormalized1stDifferenceEEG(fileprebeta1)
# print("finish5")
# X6=readNormalized1stDifferenceEEG(fileprebeta2)
# print("finish6")
# # 
# X7=readNormalized1stDifferenceEEG(filepostdelta)
# print("finish7")
# X8=readNormalized1stDifferenceEEG(filepostceta)
# print("finish8")
# X9=readNormalized1stDifferenceEEG(filepostlowalpha)
# print("finish9")
# X10=readNormalized1stDifferenceEEG(fileposthighalpha)
# print("finish10")
# X11=readNormalized1stDifferenceEEG(filepostbeta1)
# print("finish11")
# X12=readNormalized1stDifferenceEEG(filepostbeta2)
# print("finish12")
 
X13=readNormalized1stDifferenceEEG(filefudelta)
print("finish13")
X14=readNormalized1stDifferenceEEG(filefuceta)
print("finish14")
X15=readNormalized1stDifferenceEEG(filefulowalpha)
print("finish15")
X16=readNormalized1stDifferenceEEG(filefuhighalpha)
print("finish16")
X17=readNormalized1stDifferenceEEG(filefubeta1)
print("finish17")
X18=readNormalized1stDifferenceEEG(filefubeta2)
print("finish18")
X19=readNormalized1stDifferenceEEG(filepre)
print("finish19")
X20=readNormalized1stDifferenceEEG(filepost)
print("finish20")
X21=readNormalized1stDifferenceEEG(filefu)
print("finish21")

#np.savetxt('allpre_afterv6Power1-3.csv', X1, delimiter=',')
# np.save('allpre_afterv6Normalized1stDifferenceflip1-3', X1) 
# np.save('allpre_afterv6Normalized1stDifferenceflip4-7', X2) 
# np.save('allpre_afterv6Normalized1stDifferenceflip8-10', X3) 
# np.save('allpre_afterv6Normalized1stDifferenceflip10-12', X4) 
# np.save('allpre_afterv6Normalized1stDifferenceflip13-21', X5) 
# np.save('allpre_afterv6Normalized1stDifferenceflip22-30', X6) 

# np.save('allpost_afterv6Normalized1stDifferenceflip1-3', X7) 
# np.save('allpost_afterv6Normalized1stDifferenceflip4-7', X8)  
# np.save('allpost_afterv6Normalized1stDifferenceflip8-10', X9) 
# np.save('allpost_afterv6Normalized1stDifferenceflip10-12', X10) 
# np.save('allpost_afterv6Normalized1stDifferenceflip13-21', X11) 
# np.save('allpost_afterv6Normalized1stDifferenceflip22-30', X12)
 

np.save('allfu_afterv6Normalized1stDifferenceflip1-3', X13) 
np.save('allfu_afterv6Normalized1stDifferenceflip4-7', X14) 
np.save('allfu_afterv6Normalized1stDifferenceflip8-10', X15)
np.save('allfu_afterv6Normalized1stDifferenceflip10-12', X16) 
np.save('allfu_afterv6Normalized1stDifferenceflip13-21', X17) 
np.save('allfu_afterv6Normalized1stDifferenceflip22-30', X18)
np.save('allpre_afterv6Normalized1stDifferenceflip', X19) 
np.save('allpost_afterv6Normalized1stDifferenceflip', X20) 
np.save('allfu_afterv6Normalized1stDifferenceflip', X21)
print("finish")















