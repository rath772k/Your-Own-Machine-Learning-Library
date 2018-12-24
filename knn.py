#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:55:36 2018

@author: rath772k
"""
import numpy as np
class knn:
    def fit(self,x,y,k=3):
        self.x=x
        self.y=y
        self.k=k
        
    def dist(self,a,b):
        return np.sum(np.square(a-b),axis=1)
    
    def neighbours(self,xin):
        distance = self.dist(self.x,xin)
        t=np.argsort(distance,axis=0)
        neigh=self.y[t]
        neigh=neigh[:self.k,0]
        cla_count=np.bincount(neigh.astype(int))
        return np.argmax(cla_count)
    
    def normalize(self,xin):
        m = np.mean(xin,axis=0)
        s = np.std(xin,axis=0)
        return (xin-m)/s
    
    def predict(self,xin):
        m=np.shape(xin)[0]
        y_pred=np.zeros((m,1))
        for i in range(m):
            y_pred[i] = self.neighbours(xin[i,:])
        return y_pred
    
    def accuracy(self,x_test,y_test):
        return np.mean(self.predict(x_test)==y_test)*100
    
    
    
dat = np.loadtxt(open("/home/rath772k/.config/spyder-py3/DoubleMoon2.txt"),delimiter=',')
np.random.shuffle(dat)
x = dat[:,0:-1]
y = dat[:,-1:]
(a,b)=np.shape(x)
dp = knn()
dp.fit(x_train,y_train)
x = dp.normalize(x)
x_train=x[0:7*a//10,:]
y_train=y[0:7*a//10,:]
x_test=x[7*a//10:a,:]
y_test=y[7*a//10:a,:]
print(dp.accuracy(x_test,y_test))
