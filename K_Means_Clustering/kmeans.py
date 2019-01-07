#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 03:32:36 2019

@author: rath772k
"""

import numpy as np

class Kmeans:
    def fit(self,X):
        self.X=X
        self.m=np.shape(X)[0]
        
    def dist(self,a,b):
        return np.sum(np.square(a-b),axis=1)
    
    def normalize(self,xin):
        m = np.mean(xin,axis=0)
        s = np.std(xin,axis=0)
        return (xin-m)/s
    
    def cluster(self,k,no_of_iter):
        indices=np.random.choice(self.m,k,False)
        self.centroids=self.X[indices,:]
        self.cindex=np.zeros(self.m)
        for j in range(no_of_iter):
            for i in range(self.m):
                self.cindex[i]=np.argmin(self.dist(self.centroids,self.X[i]))
            for i in range(k):
                self.centroids[i]=np.mean(self.X[np.nonzero(self.cindex==i)],axis=0)
        return self.cindex

