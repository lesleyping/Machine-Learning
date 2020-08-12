#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:19:34
#
######

import os
import time

import argparse
from feature2 import *
from numpy import *
import operator


def get_feature(img_path):
    hu = get_hu(img_path)
    svd = get_svd(img_path)
    mr = get_mr(img_path)
    pzor = get_pzor(img_path)
    #sift = get_sift(img_path)
    minrectangle = get_minrectangle(img_path)
    #harris = get_harris(img_path)
    #fourier = get_fourier(img_path)

    feature = np.concatenate(( mr, pzor, minrectangle, hu, svd))
    #feature = np.concatenate(( mr, hu))
    return feature



def classify(sample , features , labels , k):
    datasatSize = len(features)

    tempSet = tile(sample,(datasatSize,1))
    tempSet = tempSet.astype('float64')
    features = np.array(features,dtype = np.float64)
    diffMat = tempSet -features
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()


    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1) , reverse = True)
    return sortedClassCount[0][0]

def decode(train_features,train_labels, test_file):
    k = 3
    right = 0
    sum = 0
    predict_labels = []
    # model {"feature" [], labels: []}

    for i, img_dir in enumerate(sorted(os.listdir(test_file))):
        img_folder = os.path.join(test_file , img_dir)
        for item in os.listdir(img_folder):
            img_path = os.path.join(img_folder , item)
            sample = get_feature(img_path)
            label = classify(sample , train_features , train_labels , k)
            sum = sum+1
            if label == i+1 :
                right = right + 1
            predict_labels.append(label)
    print(right)
    print(sum)
    correct_rate = float(right)/float(sum)

    return correct_rate,predict_labels


def train(train_dir):
    # 1
    #  -- 001.bmp
    #  -- 002.bmp
    # 2
    train_features = []
    train_labels = []

    for i, img_dir in enumerate(sorted(os.listdir(train_dir))):

        img_folder = os.path.join(train_dir, img_dir)
        for item in os.listdir(img_folder):
            img_path = os.path.join(img_folder, item)
            feature = get_feature(img_path)
            train_features.append(feature)
            train_labels.append(i+1)
    return train_features,train_labels

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    args = parser.parse_args()
    t1 = time.time()
    train_features, train_labels = train(args.train_dir)
    print("t1",time.time()-t1)
    t2 = time.time()
    c_r,predict = decode(train_features,train_labels, args.test_dir)
    print("t2", time.time() - t2)
    print(c_r)
    print(predict)
