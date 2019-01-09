#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 06:20:47 2019

@author: rath772k
"""
class DecisionTree:
    
    class question:
        def __init__(self,feature,answer):
            self.feature=feature
            self.answer=answer
            
        def match(self,row):
            value=row[self.feature]
            if isinstance(self.answer,int) or isinstance(self.answer,float):
                return value>=self.answer
            else:
                return value==self.answer
    
    #in the whole class, argument 'x' is treated as a 2 array of features
    def divide(self,qn,x,y):
        true_x,false_x,true_y,false_y=[],[],[],[]
        l=len(y)
        for i in range(l):
            if qn.match(x[i]):
                true_x.append(x[i])
                true_y.append(y[i])
            else:
                false_x.append(x[i])
                false_y.append(y[i])
        return true_x,false_x,true_y,false_y
    
    def count(self,y):
        count = {}#empty dictionary
        for value in y:
            if value not in count:
                count[value]=0
            count[value]+=1
        return count
    
    
    def gini(self,y):
        count_=self.count(y)
        impurity=1
        for value in count_:
            impurity-=(count_[value]/sum(count_.values()))**2
        return impurity
    
    def inf_gain(self,cur_imp,true_y,false_y):
        p=len(true_y)/(len(true_y)+len(false_y))
        return cur_imp-p*self.gini(true_y)-(1-p)*self.gini(false_y)
    
    def best_gain(self,x,y):
        best_gain=0
        best_qn=None
        l=len(x[0])
        for i in range(l):
            values=set(row[i] for row in x)
            for v in values:
                qn=self.question(i,v)
                true_x,false_x,true_y,false_y=self.divide(qn,x,y)
                gain=self.inf_gain(self.gini(y),true_y,false_y)
                if(gain>best_gain):
                    best_gain,best_qn=gain,qn
        
        return best_gain,best_qn
    
    class leaf:
        def __init__(self,x,y):
            self.predictions=DecisionTree().count(y)
        
    class node:
        def __init__(self,true_branch,false_branch,qn):
            self.true_branch=true_branch
            self.false_branch=false_branch
            self.qn=qn
    
           
    def build_tree(self,x,y):
        gain,qn=self.best_gain(x,y)
        
        if gain==0:
            return self.leaf(x,y)
        
        true_x,false_x,true_y,false_y=self.divide(qn,x,y)
        true_branch=self.build_tree(true_x,true_y)
        false_branch=self.build_tree(false_x,false_y)
        
        return self.node(true_branch,false_branch,qn)
    
    def fit(self,x,y):
        self.x=x #2d list of features
        self.y=y #classes of values corresponding to the features, a 1 d list 
        self.root=self.build_tree(x,y)
    
    def classify(self,x,node):#here x is a single example i.e 1 row
        if isinstance(node,self.leaf):
            return node.predictions
        if node.qn.match(x):
            return self.classify(x,node.true_branch)
        return self.classify(x,node.false_branch)
    
    def find_prob(self,x):
        l=len(x)
        answer,predict=self.predict(x)
        prob=[None]*l
        for i in range(l):
            prob[i]={}
            for key in predict[i]:
                prob[i][key]=predict[i][key]/sum(predict[i].values())
        return prob
        
        
    def predict(self,x):
        l=len(x)
        predict=[None]*l
        answer=[]
        for i in range(l):
            predict[i]=self.classify(x[i],self.root)
            listof=[(values,key) for key,values in predict[i].items()]
            answer.append(max(listof)[1])
        return answer,predict
    
    def accuracy(self,x,y):
        res,pre=self.predict(x)
        l=len(res)
        return sum([int(res[i]==y[i]) for i in range(l)])/l*100
