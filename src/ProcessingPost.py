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
filefu=[]
filepost=[]
filepre=[]
filecon=[]
for index in range (24):
    filepostname='allpost_afterv6fre'+str(index+1)+'.mat'
    fileprename='allpre_afterv6fre'+str(index+1)+'.mat'
    filefuname='allfu_afterv6fre'+str(index+1)+'.mat'
    fileconname='allcontrol_afterv6fre'+str(index+1)+'.mat'
    filefu.append(sio.loadmat(filefuname))
    filepost.append(sio.loadmat(filepostname))
    filepre.append(sio.loadmat(fileprename))
    filecon.append(sio.loadmat(fileconname))
 
 
print(filepre[0]['alldata'].shape)
print(filepost[0]['alldata'].shape)
print(filefu[0]['alldata'].shape)
print(filefu[1]['alldata'].shape)
print(filecon[0]['alldata'].shape)
print("start")
X1=readPowerEEG(filecontrol)
print("finish1")
X2=readMeanEEG(filecontrol)
print("finish2")
X3=readStandardDeviationEEG(filecontrol)
print("finish3")
X4=read1stDifferenceEEG(filecontrol)
print("finish4")
X5=readNormalized1stDifferenceEEG(filecontrol)
print("finish5")
X6=read2stDifferenceEEG(filecontrol)
print("finish6")
X7=readNormalized2stDifferenceEEG(filecontrol)
print("finish7")
 
np.savetxt('allpost_afterv6Power.csv', X1, delimiter=',')
np.save('allpost_afterv6Power', X1) 
np.savetxt('allpost_afterv6Mean.csv', X2, delimiter=',')
np.save('allpost_afterv6Mean', X2)
np.savetxt('allpost_afterv6StandardDeviation.csv', X3, delimiter=',')
np.save('allpost_afterv6StandardDeviation', X3) 
np.savetxt('allpost_afterv61stDifference.csv', X4, delimiter=',')
np.save('allpost_afterv61stDifference', X4) 
np.savetxt('allpost_afterv6Normalized1stDifference.csv', X5, delimiter=',')
np.save('allpost_afterv6Normalized1stDifference', X5) 
np.savetxt('allpost_afterv62stDifference.csv', X6, delimiter=',')
np.save('allpost_afterv62stDifference', X6) 
np.savetxt('allpost_afterv6Normalized2stDifference.csv', X7, delimiter=',')
np.save('allpost_afterv6Normalized2stDifference', X7) 
 


print("finish")

fileresp=sio.loadmat('allresp.mat')
percentb=np.zeros(shape=[24,1])
percentg=np.zeros(shape=[24,1])
percentw=np.zeros(shape=[24,1])
for index in range(0,24):
    percentb[index]=fileresp['allresp'][0][index][1][0]/441
    percentg[index]=fileresp['allresp'][0][index][1][1]/441
    percentw[index]=fileresp['allresp'][0][index][1][2]/441
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
np.savetxt('allpost_afterv6respB.csv', percentControlb, delimiter=',')
np.save('allpost_afterv6respB', percentControlb) 
np.savetxt('allpost_afterv6respG.csv', percentControlg, delimiter=',')
np.save('allpost_afterv6respG', percentControlg) 
np.savetxt('allpost_afterv6respW.csv', percentControlw, delimiter=',')
np.save('allpost_afterv6respW', percentControlw) 
np.savetxt('allpost_afterv6resp.csv', percentControlbgw, delimiter=',')
np.save('allpost_afterv6resp', percentControlbgw) 




##25 area label


Area25=np.load('To_25_post.npy')
 
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
  
Area25Post=np.zeros(shape=[len(filepost['alldata2']),25])
for index in range(len(filepost['alldata2'])):
    number=filepost['alldata2'][index][1][0][0]-1
    for index2 in range(25):
        Area25Post[index][index2]=Area25[number][index2]
np.save('allpost_afterv6resp25area', Area25Post) 
print('finish post')
  
Area25=np.load('To_25_pre.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
Area25Pre=np.zeros(shape=[len(filepre['alldata2']),25])
for index in range(len(filepre['alldata2'])):
    number=filepre['alldata2'][index][1][0][0]-1
    for index2 in range(25):
        Area25Pre[index][index2]=Area25[number][index2]
np.save('allpre_afterv6resp25area', Area25Pre) 
  
print('finish pre')
  
Area25=np.load('To_25_fu.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
Area25Fu=np.zeros(shape=[len(filefu['alldata2']),25])
for index in range(len(filefu['alldata2'])):
    number=filefu['alldata2'][index][1][0][0]-1
    for index2 in range(25):
        Area25Fu[index][index2]=Area25[number][index2]
np.save('allfu_afterv6resp25area', Area25Fu) 
  
print('finish fu')
  
Area25Con=np.zeros(shape=[3764,25])
for index in range(3764):
    for index2 in range(25):
        Area25Con[index][index2]=1
np.save('allcontrol_afterv6resp25area', Area25Con) 
  
print('finish con')

##25 area label seperate
 
 
Area25=np.load('To_25_post.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
for index3 in range(24):
    Area25Post=np.zeros(shape=[len(filepost[index3]['alldata']),25])
    for index in range(len(filepost[index3]['alldata'])):
        number=index3
        for index2 in range(25):
            Area25Post[index][index2]=Area25[number][index2]
    filename='allpost_afterv6resp25area'+str(index3+1)
    np.save(filename, Area25Post) 
print('finish post')
    
Area25=np.load('To_25_pre.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
for index3 in range(24):
    Area25Pre=np.zeros(shape=[len(filepre[index3]['alldata']),25])
    for index in range(len(filepre[index3]['alldata'])):
        number=index3
        for index2 in range(25):
            Area25Pre[index][index2]=Area25[number][index2]
    filename='allpre_afterv6resp25area'+str(index3+1)
    np.save(filename, Area25Pre) 
print('finish pre')
    
Area25=np.load('To_25_fu.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
for index3 in range(24):
    Area25Fu=np.zeros(shape=[len(filefu[index3]['alldata']),25])
    print(Area25Fu.shape)
    for index in range(len(filefu[index3]['alldata'])):
        number=index3
        for index2 in range(25):
            Area25Fu[index][index2]=Area25[number][index2]
    filename='allfu_afterv6resp25area'+str(index3+1)
    np.save(filename, Area25Fu) 
    
print('finish fu')
    
 
for index3 in range(24):
    Area25Con=np.zeros(shape=[len(filecon[index3]['alldata']),25])
    for index in range(len(filecon[index3]['alldata'])):
        for index2 in range(25):
            Area25Con[index][index2]=1
    filename='allcontrol_afterv6resp25area'+str(index3+1)
    np.save(filename, Area25Con) 
print('finish con')
##4 area label seperate
 
  
Area4=np.load('To_4_post.npy')
for index in range(len(Area4)):
    for index2 in range(4):
        if Area4[index][index2]!=0:
            Area4[index][index2]=Area4[index][index2]-1
for index3 in range(24):
    Area4Post=np.zeros(shape=[len(filepost[index3]['alldata']),4])
    for index in range(len(filepost[index3]['alldata'])):
        number=index3
        for index2 in range(4):
            Area4Post[index][index2]=Area4[number][index2]
    filename='allpost_afterv6resp4area'+str(index3+1)
    print(Area4Post.shape)
    np.save(filename, Area4Post) 
print('finish post')
    
Area4=np.load('To_4_pre.npy')
for index in range(len(Area4)):
    for index2 in range(4):
        if Area4[index][index2]!=0:
            Area4[index][index2]=Area4[index][index2]-1
for index3 in range(24):
    Area4Pre=np.zeros(shape=[len(filepre[index3]['alldata']),4])
    for index in range(len(filepre[index3]['alldata'])):
        number=index3
        for index2 in range(4):
            Area4Pre[index][index2]=Area4[number][index2]
    filename='allpre_afterv6resp4area'+str(index3+1)
    np.save(filename, Area4Pre) 
print('finish pre')
    
Area4=np.load('To_4_fu.npy')
for index in range(len(Area4)):
    for index2 in range(4):
        if Area4[index][index2]!=0:
            Area4[index][index2]=Area4[index][index2]-1
for index3 in range(24):
    Area4Fu=np.zeros(shape=[len(filefu[index3]['alldata']),4])
    for index in range(len(filefu[index3]['alldata'])):
        number=index3
        for index2 in range(4):
            Area4Fu[index][index2]=Area4[number][index2]
    filename='allfu_afterv6resp4area'+str(index3+1)
    np.save(filename, Area4Fu) 
    
print('finish fu')
    
 
for index3 in range(24):
    Area4Con=np.zeros(shape=[len(filecon[index3]['alldata']),4])
    for index in range(len(filecon[index3]['alldata'])):
        for index2 in range(4):
            Area4Con[index][index2]=1
    filename='allcontrol_afterv6resp4area'+str(index3+1)
    np.save(filename, Area4Con) 
print('finish con')

##4 area label

fourArea=np.load('To_4_post.npy')
  
for index in range(len(fourArea)):
    for index2 in range(4):
        if fourArea[index][index2]!=0:
            fourArea[index][index2]=fourArea[index][index2]-1
  
fourAreaPost=np.zeros(shape=[len(filepost['alldata2']),4])
for index in range(len(filepost['alldata2'])):
    number=filepost['alldata2'][index][1][0][0]-1
    for index2 in range(4):
        fourAreaPost[index][index2]=fourArea[number][index2]
np.save('allpost_afterv6resp4area', fourAreaPost) 
print('finish post')
  
fourArea=np.load('To_4_pre.npy')
for index in range(len(fourArea)):
    for index2 in range(4):
        if fourArea[index][index2]!=0:
            fourArea[index][index2]=fourArea[index][index2]-1
fourAreaPre=np.zeros(shape=[len(filepre['alldata2']),4])
for index in range(len(filepre['alldata2'])):
    number=filepre['alldata2'][index][1][0][0]-1
    for index2 in range(4):
        fourAreaPre[index][index2]=fourArea[number][index2]
np.save('allpre_afterv6resp4area', fourAreaPre) 
  
print('finish pre')
  
fourArea=np.load('To_4_fu.npy')
for index in range(len(fourArea)):
    for index2 in range(4):
        if fourArea[index][index2]!=0:
            fourArea[index][index2]=fourArea[index][index2]-1
fourAreaFu=np.zeros(shape=[len(filefu['alldata2']),4])
for index in range(len(filefu['alldata2'])):
    number=filefu['alldata2'][index][1][0][0]-1
    for index2 in range(4):
        fourAreaFu[index][index2]=fourArea[number][index2]
np.save('allfu_afterv6resp4area', fourAreaFu) 
  
print('finish fu')
  
  
fourAreaCon=np.zeros(shape=[3764,4])
for index in range(3764):
    for index2 in range(4):
        fourAreaCon[index][index2]=1
np.save('allcontrol_afterv6resp4area', fourAreaCon) 
  
print('finish con')

##16 area label seperate
 
  
Area16=np.load('To_16_post.npy')
for index in range(len(Area16)):
    for index2 in range(16):
        if Area16[index][index2]!=0:
            Area16[index][index2]=Area16[index][index2]-1
for index3 in range(24):
    Area16Post=np.zeros(shape=[len(filepost[index3]['alldata']),16])
    for index in range(len(filepost[index3]['alldata'])):
        number=index3
        for index2 in range(16):
            Area16Post[index][index2]=Area16[number][index2]
    filename='allpost_afterv6resp16area'+str(index3+1)
    np.save(filename, Area16Post) 
print('finish post')
    
Area16=np.load('To_16_pre.npy')
for index in range(len(Area16)):
    for index2 in range(16):
        if Area16[index][index2]!=0:
            Area16[index][index2]=Area16[index][index2]-1
for index3 in range(24):
    Area16Pre=np.zeros(shape=[len(filepre[index3]['alldata']),16])
    for index in range(len(filepre[index3]['alldata'])):
        number=index3
        for index2 in range(16):
            Area16Pre[index][index2]=Area16[number][index2]
    filename='allpre_afterv6resp16area'+str(index3+1)
    np.save(filename, Area16Pre) 
print('finish pre')
    
Area16=np.load('To_16_fu.npy')
for index in range(len(Area16)):
    for index2 in range(16):
        if Area16[index][index2]!=0:
            Area16[index][index2]=Area16[index][index2]-1
for index3 in range(24):
    Area16Fu=np.zeros(shape=[len(filefu[index3]['alldata']),16])
    for index in range(len(filefu[index3]['alldata'])):
        number=index3
        for index2 in range(16):
            Area16Fu[index][index2]=Area16[number][index2]
    filename='allfu_afterv6resp16area'+str(index3+1)
    np.save(filename, Area16Fu) 
    
print('finish fu')
    
 
for index3 in range(24):
    Area16Con=np.zeros(shape=[len(filecon[index3]['alldata']),16])
    for index in range(len(filecon[index3]['alldata'])):
        for index2 in range(16):
            Area16Con[index][index2]=1
    filename='allcontrol_afterv6resp16area'+str(index3+1)
    np.save(filename, Area16Con) 
print('finish con')

##9 area label seperate



 
Area9=np.load('To_9_post.npy')
for index in range(len(Area9)):
    for index2 in range(9):
        if Area9[index][index2]!=0:
            Area9[index][index2]=Area9[index][index2]-1
for index3 in range(24):
    Area9Post=np.zeros(shape=[len(filepost[index3]['alldata']),9])
    for index in range(len(filepost[index3]['alldata'])):
        number=index3
        for index2 in range(9):
            Area9Post[index][index2]=Area9[number][index2]
    filename='allpost_afterv6resp9area'+str(index3+1)
    np.save(filename, Area9Post) 
print('finish post')
    
Area9=np.load('To_9_pre.npy')
for index in range(len(Area9)):
    for index2 in range(9):
        if Area9[index][index2]!=0:
            Area9[index][index2]=Area9[index][index2]-1
for index3 in range(24):
    Area9Pre=np.zeros(shape=[len(filepre[index3]['alldata']),9])
    for index in range(len(filepre[index3]['alldata'])):
        number=index3
        for index2 in range(9):
            Area9Pre[index][index2]=Area9[number][index2]
    filename='allpre_afterv6resp9area'+str(index3+1)
    np.save(filename, Area9Pre) 
print('finish pre')
    
Area9=np.load('To_9_fu.npy')
for index in range(len(Area9)):
    for index2 in range(9):
        if Area9[index][index2]!=0:
            Area9[index][index2]=Area9[index][index2]-1
for index3 in range(24):
    Area9Fu=np.zeros(shape=[len(filefu[index3]['alldata']),9])
    for index in range(len(filefu[index3]['alldata'])):
        number=index3
        for index2 in range(9):
            Area9Fu[index][index2]=Area9[number][index2]
    filename='allfu_afterv6resp9area'+str(index3+1)
    np.save(filename, Area9Fu) 
    
print('finish fu')
    
 
for index3 in range(24):
    Area9Con=np.zeros(shape=[len(filecon[index3]['alldata']),9])
    for index in range(len(filecon[index3]['alldata'])):
        for index2 in range(9):
            Area9Con[index][index2]=1
    filename='allcontrol_afterv6resp9area'+str(index3+1)
    np.save(filename, Area9Con) 
print('finish con')

##percentage label seperate
fileresp=sio.loadmat('allresp.mat')
percentPre=np.zeros(shape=[24,3])
percentPost=np.zeros(shape=[24,3])
percentFu=np.zeros(shape=[24,3])
print(fileresp['allresp'][0][1])
for index in range(0,24):
    for index2 in range(0,3):
        percentPre[index][index2]=fileresp['allresp'][0][index][0][index2]/441
        percentPost[index][index2]=fileresp['allresp'][0][index][1][index2]/441
        percentFu[index][index2]=fileresp['allresp'][0][index][2][index2]/441
 
print(percentPost[1])
print(percentPost[1][0])
print(len(filepost))
 
for index3 in range(24):
    newpercentPre=np.zeros(shape=[len(filepre[index3]['alldata']),3])
    for index in range(len(filepre[index3]['alldata'])):
        number=index3
        for index2 in range(3):
            newpercentPre[index][index2]=percentPre[number][index2]
    filename='allpre_afterv6resppercent'+str(index3+1)
    np.save(filename, newpercentPre) 
print('finish pre')
 
for index3 in range(24):
    newpercentPost=np.zeros(shape=[len(filepost[index3]['alldata']),3])
    for index in range(len(filepost[index3]['alldata'])):
        number=index3
        print(number)
        for index2 in range(3):
            newpercentPost[index][index2]=percentPost[number][index2]
    filename='allpost_afterv6resppercent'+str(index3+1)
    np.save(filename, newpercentPost) 
print('finish post')
# print(newpercentPost.shape)
      
      
  
for index3 in range(24):
    newpercentFu=np.zeros(shape=[len(filefu[index3]['alldata']),3])
    for index in range(len(filefu[index3]['alldata'])):
        number=index3
        for index2 in range(3):
            newpercentFu[index][index2]=percentFu[number][index2]
    filename='allfu_afterv6resppercent'+str(index3+1)
    np.save(filename, newpercentFu) 
       
print('finish fu')
      
   
for index3 in range(24):
    percentCon=np.zeros(shape=[len(filecon[index3]['alldata']),3])
    for index in range(len(filecon[index3]['alldata'])):
        number=index3
        percentCon[index][0]=0
        percentCon[index][1]=0
        percentCon[index][2]=1
    filename='allcontrol_afterv6resppercent'+str(index3+1)
    np.save(filename, percentCon) 
print('finish con')


##4 area label seperate connectivity
Area4=np.load('To_4_post.npy')
for index in range(len(Area4)):
    for index2 in range(4):
        if Area4[index][index2]!=0:
            Area4[index][index2]=Area4[index][index2]-1
for index3 in range(24):
    Area4Post=np.zeros(shape=[10,4])
    for index in range(10):
        number=index3
        for index2 in range(4):
            Area4Post[index][index2]=Area4[number][index2]
    filename='allpost_afterv6resp4area10connectivity'+str(index3+1)
    np.save(filename, Area4Post) 
print('finish post')
      
Area4=np.load('To_4_pre.npy')
for index in range(len(Area4)):
    for index2 in range(4):
        if Area4[index][index2]!=0:
            Area4[index][index2]=Area4[index][index2]-1
for index3 in range(24):
    Area4Pre=np.zeros(shape=[10,4])
    for index in range(10):
        number=index3
        for index12 in range(4):
            Area4Pre[index][index2]=Area4[number][index2]
    filename='allpre_afterv6resp4area10connectivity'+str(index3+1)
    np.save(filename, Area4Pre) 
print('finish pre')
       
Area4=np.load('To_4_fu.npy')
for index in range(len(Area4)):
    for index2 in range(4):
        if Area4[index][index2]!=0:
            Area4[index][index2]=Area4[index][index2]-1
for index3 in range(24):
    Area4Fu=np.zeros(shape=[10,4])
    for index in range(10):
        number=index3
        for index2 in range(4):
            Area4Fu[index][index2]=Area4[number][index2]
    filename='allfu_afterv6resp4area10connectivity'+str(index3+1)
    np.save(filename, Area4Fu) 
       
print('finish fu')
       
    
for index3 in range(24):
    Area4Con=np.zeros(shape=[10,4])
    for index in range(10):
        for index2 in range(4):
            Area4Con[index][index2]=1
    filename='allcontrol_afterv6resp4area10connectivity'+str(index3+1)
    np.save(filename, Area4Con) 
print('finish con')

##9 area label seperate connectivity
Area25=np.load('To_25_post.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
for index3 in range(24):
    Area25Post=np.zeros(shape=[10,25])
    for index in range(10):
        number=index3
        for index2 in range(25):
            Area25Post[index][index2]=Area25[number][index2]
    filename='allpost_afterv6resp25area10connectivity'+str(index3+1)
    np.save(filename, Area25Post) 
print('finish post')
      
Area25=np.load('To_25_pre.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
for index3 in range(24):
    Area25Pre=np.zeros(shape=[10,25])
    for index in range(10):
        number=index3
        for index12 in range(25):
            Area25Pre[index][index2]=Area25[number][index2]
    filename='allpre_afterv6resp25area10connectivity'+str(index3+1)
    np.save(filename, Area25Pre) 
print('finish pre')
       
Area25=np.load('To_25_fu.npy')
for index in range(len(Area25)):
    for index2 in range(25):
        if Area25[index][index2]!=0:
            Area25[index][index2]=Area25[index][index2]-1
for index3 in range(24):
    Area25Fu=np.zeros(shape=[10,25])
    for index in range(10):
        number=index3
        for index2 in range(25):
            Area25Fu[index][index2]=Area25[number][index2]
    filename='allfu_afterv6resp25area10connectivity'+str(index3+1)
    np.save(filename, Area25Fu) 
       
print('finish fu')
       
    
for index3 in range(24):
    Area25Con=np.zeros(shape=[10,25])
    for index in range(10):
        for index2 in range(25):
            Area25Con[index][index2]=1
    filename='allcontrol_afterv6resp25area10connectivity'+str(index3+1)
    np.save(filename, Area25Con) 
print('finish con')