import random
import os
import sys
import math
import numpy as np
import copy

from PIL import Image



trainPath = "D://python2.7.6//MachineLearning//pcawhiten//trainingDigits"
datafile="D://python2.7.6//MachineLearning//pcawhiten//dataVec.txt"
labelfile="D://python2.7.6//MachineLearning//pcawhiten//dataLabel.txt" 
 
     

global classDic;classDic={}
global allLabel;allLabel=[]
global classList;classList=[]
global dataMat
 

######################

def loadData():
    global dim,dd
    global dataMat,classDic,allLabel,dataList
    dataList=[]
    ###################labellist
    for filename in os.listdir(trainPath):
        pos=filename.find('_')
        clas=int(filename[:pos])
        if clas not in classDic:classDic[clas]=1.0
        else:classDic[clas]+=1.0
        allLabel.append(clas)
    ###########
    for filename in os.listdir(trainPath):
        obs=[]
        content=open(trainPath+'/'+filename,'r')
        line=content.readline().strip('\n')
        while len(line)!=0:
            hang=[]
            for num in line:
                hang.append(float(num))
            obs.append(hang)
            line=content.readline().strip('\n')
        ########obs list32x32 ->mat  32x32 
        obsMat=np.mat(obs)
        flip=random.sample([1,2,3,4,0],1)[0]# transpose /mirror horizen /mirror verticle/horizen and verticle
        ###mirror before half dim
        mirrorMat=rotate(obsMat,0)# 32x32 ->1 x 32x32 
        newMat=halfDim(mirrorMat)#32x32 obs ->16x16 obs
        ###########mat->list
        dataList.append(list(newMat.A.flatten())) #dataMat n x 32x32
    print 'total image %d with dim %d'%(len(dataList),len(dataList[0])) #len(array)!=len(list)
    ##############
    dataMat=np.mat(dataList)
    n,d=np.shape(dataMat)
    outPutfile=open(datafile,'w')
     
    for i in range(n):
        for j in range(d):
            outPutfile.write(str(dataMat[i,j]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    ######
    outPutfile=open(labelfile,'w')
     
    for label in allLabel:
        outPutfile.write(str(label))
        outPutfile.write(' ')
         
    outPutfile.close()   
            

############################support
def rotate(obsMat,flip): #32x32
    m,n=np.shape(obsMat)
    #######
    if flip==1:
        mirrorMat=np.mat(np.zeros((m,n)))
        mirrorMat=obsMat.T
    elif flip==2:
        mirrorMat=np.mat(np.zeros((m,n)))
        for i in range(m):
            for j in range(n):
                mirrorMat[i,j]=obsMat[m-i-1,j]
    elif flip==3:
        mirrorMat=np.mat(np.zeros((m,n)))
        for i in range(m):
            for j in range(n):
                mirrorMat[i,j]=obsMat[i,n-1-j]
    elif flip==4:
        mirrorMat=np.mat(np.zeros((m,n)))
        for i in range(m):
            for j in range(n):
                mirrorMat[i,j]=obsMat[m-1-i,n-1-j]
    ####################
    elif flip==0:
        mirrorMat=obsMat
 
    return mirrorMat


def halfDim(obsMat):
    m0,n0=np.shape(obsMat)
    m1=m0/2;n1=n0/2
    newMat=np.mat(np.zeros((m1,n1)))#16x16
    pwind=2
    for i in range(m1):
        for j in range(n1):
            patch=obsMat[i*pwind:i*pwind+pwind,j*pwind:j*pwind+pwind]
            newMat[i,j]=patch.sum(0).sum(1)[0,0]
    ############
    return newMat
    
    
    
    
    
#############
loadData()





 
 

 
    
         
    
    
    







    
    
