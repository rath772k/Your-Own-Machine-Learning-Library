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
            self.w[i]=np.random.rand(self.ls[i],self.ls[i+1])*2*0.12-0.12
            self.b[i]=np.random.rand(1,self.ls[i+1])*2*0.12-0.12
            
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
        cost2=0
        cost1 = np.sum(self.y.dot(np.log(self.act[self.nol-1]))+(1-self.y).dot(np.log(1-self.act[self.nol-1])))
        for i in range(self.nol):
            cost2 += np.sum(np.square(self.w[i]))
            return 1/self.m*(cost2/2-cost1)
        
    def backpropagation(self):
        deltaw=[None]*(self.nol-1)
        deltab=[None]*(self.nol-1)
        delt = [None]*self.nol
        self.feedforward(self.X)
        delt[self.nol-1]=self.act[self.nol-1]-self.y
        for j in range(self.nol-2,0,-1):
            delt[j]=delt[j+1].dot(self.w[j].T)*self.act[j]*(1-self.act[j])
        for j in range(self.nol-1):
            deltaw[j]=(np.dot(self.act[j].T,delt[j+1]))/self.m+(self.lambda_*self.w[j])/self.m
            deltab[j]=(np.sum(delt[j+1],axis=0))/self.m
            self.w[j]-=self.l_rate*deltaw[j]
            self.b[j]-=self.l_rate*deltab[j]
            
    def grad_descent(self,l_rate,no_of_iter):
        self.l_rate=l_rate
        for i in range(no_of_iter):
            self.backpropagation()
        return self.w,self.b
    
    def predict(self,xin):
        self.feedforward(xin)
        res=self.classes[np.argmax(self.act[self.nol-1],axis=1)]
        return res
    
    def accuracy(self,x,y):
        res=self.predict(x)
        return np.mean(y==res)*100


                
        
