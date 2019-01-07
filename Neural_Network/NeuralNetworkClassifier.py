#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 01:31:14 2019

@author: rath772k
"""
import numpy as np
class NNClassifier:
    def fit(self,X,Y,hs,lambda_=0):
        self.m=np.size(Y)
        self.X=X   #features of 'm' training data
        self.Y=Y   #classes of 'm' training data
        self.lambda_=lambda_ #regularization factor
        """converting each unique class to the final neural network layer"""
        self.classes=np.unique(Y)
        self.noc=self.classes.size
        self.y=np.zeros((self.m,self.noc))
        for i in range(self.m):
              self.y[i,np.nonzero(self.classes==Y[i])]=1
              
        hz=np.array(hs)
        self.ls=np.insert(hz,[0,hz.size],[np.shape(X)[1],self.classes.size])
        self.nol=np.size(self.ls)
        #sizes of each layer of the neural network
                
        self.w=[None]*(self.nol-1)
        self.b=[None]*(self.nol-1)
        
        for i in range(self.nol-1):
            self.w[i]=np.random.rand(self.ls[i],self.ls[i+1])
            self.b[i]=np.random.rand(1,self.ls[i+1])
            
        self.act=[None]*self.nol
            
    def normalize(self,xin):
        Mean=np.mean(xin,axis=0)
        Std=np.std(xin,axis=0)
        return (xin - Mean ) / Std
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def feedforward(self,x):
        self.act[0]=x
        for i in range(1,self.nol):
            self.act[i]=self.sigmoid(np.dot(self.act[i-1],self.w[i-1])+self.b[i-1])
            
    def cost_fn(self):
        cost1,cost2=0,0
        for i in range(self.m):
            cost1 += np.sum(self.y[i:i+1].dot(np.log(self.act[self.nol-1]))+(1-self.y[i:i+1]).dot(np.log(1-self.act[self.nol-1])))
        for i in range(self.nol):
            cost2 += np.sum(np.square(self.w[i]))
            return 1/self.m*(cost2/2-cost1)
        
    def backpropagation(self):
        deltaw=[None]*(self.nol-1)
        deltab=[None]*(self.nol-1)
        delt = [None]*self.nol
        self.feedforward(self.X)
        delt[self.nol-1]=self.act[self.nol-1]-self.y
        for j in range(self.nol-2,-1,-1):
            delt[j]=(self.w[j].dot(delt[j+1].T).T)*self.act[j]*(1-self.act[j])
        for j in range(self.nol-1):
            deltaw[j]=(delt[j+1].T.dot(self.act[j]).T+self.lambda_*self.w[j])/self.m
            deltab[j]=(np.sum(delt[j+1],axis=0))/self.m
        return deltaw,deltab
    
    def grad_descent(self,l_rate,no_of_iter):
        for i in range(no_of_iter):
            deltaw,deltab=self.backpropagation()
            for j in range(self.nol-1):
                self.w[j]-=l_rate*deltaw[j]
                self.b[j]-=l_rate*deltab[j]
        return self.w,self.b
    
    def predict(self,xin):
        self.feedforward(xin)
        res=self.classes[np.argmax(self.act[self.nol-1],axis=1)]
        return res
    
    def accuracy(self,x,y):
        res=self.predict(x)
        return np.mean(y==res)*100
    
dat = np.loadtxt(open("/home/rath772k/.config/spyder-py3/DoubleMoon2.txt"),delimiter=',')
np.random.shuffle(dat)
x = dat[:,0:-1]
y = dat[:,-1:]
x=np.array(x)
(a,b)=np.shape(x)
dp = NNClassifier()

x = dp.normalize(x)
x_train=x[0:7*a//10,:]
y_train=y[0:7*a//10,:]
x_test=x[7*a//10:a,:]
y_test=y[7*a//10:a,:]
dp.fit(x_train,y_train,[9])
dp.grad_descent(0.001,100)
print(dp.accuracy(x_test,y_test))

from sklearn.neural_network import MLPClassifier
ap=MLPClassifier(hidden_layer_sizes=(3), activation='logistic', solver='lbfgs', alpha=0, learning_rate_init=0.001, max_iter=200)
ap.fit(x_train,y_train)
res=ap.predict(x_test)
print(np.mean(y_test==res)*100)

                
        
