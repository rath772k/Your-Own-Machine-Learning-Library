#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

#class that models linear regression
class lin_reg:
    def fit(self,x,y,lambda_=0):
        self.x = np.array(x)
        self.y = np.y
        self.w = np.zeros(np.shape(x)[1])  #weights
        self.b = [0]                       #bias
        self.lambda_ = lambda_             #regularization factor 
     
    def normalize(self,xin):
        Mean=np.mean(xin,axis=0)
        Std=np.std(xin,axis=0)
        return (xin - Mean ) / Std,Mean,Std
    
    def cost_fn(self)
        return np.sum(np.square(self.x.dot(self.w)+self.b-self.y))/2/np.size(self.y)+lambda_/2/np.size(self.y)*np.square(self.w)
    
    def grad_descent(self,l_rate,no_of_iter,lambda_):
        m=np.size(self.y)
        for i in range(no_of_iter)
            self.w -= l_rate / m * self.x.T.dot(self.x.dot(self.w) + self.b - self.y) + lambda_ / m * self.w
            self.b -= np.sum(l_rate / m * (self.x.dot(self.w) + self.b - self.y))
        return self.w,self.b
    
    def predict(self,xin)
        return np.dot(xin,self.w)+self.b
    
    def accuracy(self,x,y)
        y_pred = self.predict(x)
        err = np.abs(np.sum((y_pred-y)/y))
        return (1-err)*100
    


# In[ ]:





# In[ ]:




