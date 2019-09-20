#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:19:34
#
######

import time
import matplotlib.pyplot as plt
import pylab
import numpy as np
import random
from scipy import linalg as sp
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree

def RF_train(X_train, Y_train, d, n):
    clf=[]
    for i in range(n):
        idx=np.random.choice(len(Y_train),len(Y_train),replace=True)
        x=X_train[idx,:]
        y=Y_train[idx]
        temp=tree.DecisionTreeClassifier(max_depth=d)
        clf.append(temp.fit(x,y))
    return clf

def RF_test(clf, X_test, l, n):
    res=[]
    for j in range(l):
        r=[0,0,0,0,0,0]
        for i in range(n):
            temp=clf[i].predict(X_test[j,:])
            r[int(temp[0])]+=1
        idx=r.index(max(r))
        res.append(idx)
    return res

def OOB_error(X_train, Y_train, d):
    clf=[]
    err=[]
    n=np.round(np.linspace(1,200,num=10))
    for k in n:
        res=[]
        for i in range(int(k)):
            idx=np.random.choice(len(Y_train),len(Y_train),replace=True)
            x=X_train[idx,:]
            y=Y_train[idx]
            temp=tree.DecisionTreeClassifier(max_depth=d)
            clf.append(temp.fit(x,y))
            idx=np.unique(idx)
            oob=[]
            for j in range(len(Y_train)):
                if j not in idx:
                    oob.append(j)
            r=[0,0,0,0,0,0]
            for j in oob:
                temp=clf[i].predict(X_train[j,:])
                r[int(temp[0])]+=1
            idx=r.index(max(r))
            res.append(idx)
        temp=len([i for i, j in zip(res,Y_train[oob]) if i!=j])
        k=int(k)
        err.append(temp/k)
        print(k)
    plt.figure(5, figsize=(8,6))
    plt.plot(n,err)
    plt.ylabel('OOB Error')
    plt.xlabel('Number of Trees (Max Depth=full)')
    plt.show()

def Train_Test_err(X_train,Y_train,X_test,Y_test):
    n=np.round(np.linspace(1,500,num=10))
    err1=[]
    err2=[]
    err3=[]
    err4=[]
    for i in n:
        clf=RF_train(X_train, Y_train, len(Y_train), int(i))
        res=RF_test(clf,X_train,len(Y_train), int(i))
        temp=len([i for i, j in zip(res,Y_train) if i!=j])
        err1.append(temp/len(Y_train))
        res=RF_test(clf,X_test,len(Y_test), int(i))
        temp=len([i for i, j in zip(res,Y_test) if i!=j])
        err3.append(temp/len(Y_test))

        clf=RF_train(X_train, Y_train, 5, int(i))
        res=RF_test(clf,X_train,len(Y_train), int(i))
        temp=len([i for i, j in zip(res,Y_train) if i!=j])
        err2.append(temp/len(Y_train))
        res=RF_test(clf,X_test,len(Y_test), int(i))
        temp=len([i for i, j in zip(res,Y_test) if i!=j])
        err4.append(temp/len(Y_test))

    plt.figure(1, figsize=(8,6))
    plt.plot(n,err1)
    plt.ylabel('Train Error')
    plt.xlabel('Number of Trees (Max Depth=full)')
    plt.figure(2, figsize=(8,6))
    plt.plot(n,err2)
    plt.ylabel('Train Error')
    plt.xlabel('Number of Trees (Max Depth=5)')
    plt.figure(3, figsize=(8,6))
    plt.plot(n,err3)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees (Max Depth=full)')
    plt.figure(4, figsize=(8,6))
    plt.plot(n,err4)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees (Max Depth=5)')
    plt.show()

def Adaboost_train(X_train, Y_train, d, num_classes):
    w=[]
    for i in range(len(Y_train)):
        w.append(1/len(Y_train))
    alpha=[]
    for i in range(20):
        alpha.append(0)
    err=[]
    for i in range(20):
        err.append(0)
    clf=[]
    for i in range(20):
        x=X_train
        y=Y_train
        temp=tree.DecisionTreeClassifier(max_depth=d)
        clf.append(temp.fit(x,y,sample_weight=w))
        t=0
        for j in range(len(Y_train)):
            pred=clf[i].predict(x[j,:])
            if pred[0]!=y[j]:
                t=t+w[j]
        err[i]=t/np.sum(w)
        if err[i]==0:
            alpha[i]=np.log(num_classes-1)
        else:
            alpha[i]=np.log((1-err[i])/err[i]) + np.log(num_classes-1)
        for j in range(len(Y_train)):
            pred=clf[i].predict(x[j,:])
            if pred[0]!=y[j]:
                w[j]=w[j]*np.exp(alpha[i])
        w=w/np.sum(w)
    return alpha,clf

def Adaboost_test(clf, alpha, X_test, l, n):
    Y=[]
    classes=[1,2,3,4]
    for i in range(l):
        Y.append(0)
    for i in range(l):
        c=[0,0,0,0]
        for j in range(n):
            for k in range(len(alpha)):
                pred=clf[k].predict(X_test[i,:])
                if pred[0]==classes[j]:
                    c[j]=c[j]+alpha[k]
        Y[i]=classes[int(c.index(max(c)))]
    return Y

train_dataset = np.mat(np.loadtxt('4_train.txt', delimiter=","))
test_dataset = np.mat(np.loadtxt('4_test.txt',delimiter=","))
print(train_dataset[0])
x_train = train_dataset[:, 0:23]
y_train = train_dataset[:, 23]
x_test = test_dataset[:, 0:23]
y_test = test_dataset[:, 23]

print(len(x_train))
print(len(y_train))

#train
# mnist = np.mat(np.loadtxt("train.csv", delimiter=","))
# mnist=mnist.T
# print(mnist.shape)
# X_train=mnist[:,0:784]
# Y_train=mnist[:,784]
num_trees=20
t3 = time.time()
clf=RF_train(x_train, y_train, 5, num_trees)
print("t3:", time.time()-t3)
OOB_error(x_train, y_train, len(y_train))
t4 = time.time()
res=RF_test(clf,x_test,len(y_test), num_trees)
print("t4:",time.time()-t4)
target_names = ['1', '2', '3', '4']
print(classification_report(y_test, res, target_names=target_names))
cm=confusion_matrix(y_test, res)
print(cm)

Train_Test_err(x_train,y_train,x_test,y_test)
t1 = time.time()
alpha,clf=Adaboost_train(x_train, y_train, 10, 4)
print("t1",time.time()-t1)
print(alpha)
t2 = time.time()
Y_pred=Adaboost_test(clf,alpha,x_test,len(y_test),4)
print("t2:",time.time()-t2)
# target_names = ['1', '2', '3', '4', '5']
target_names = ['1', '2', '3', '4']
print(classification_report(y_test, Y_pred, target_names=target_names))
cm=confusion_matrix(y_test, Y_pred)
print(cm)