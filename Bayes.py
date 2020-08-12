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
import csv
import random
import math
from feature2 import *


def maxmin_normalization(feature):
    return [(float(i) - float(min(feature)))/float(float(max(feature)) - float(min(feature))) for i in feature]

def standard_normalization(feature):
    mean = sum([float(x) for x in feature]) / float(len(feature))
    variance = sum([pow(float(x) - mean , 2) for  x in feature]) / float(len(feature) -1)
    stdev = math.sqrt(variance)


    return [((float(i) - mean) / stdev) for i in feature]

def log_normalization(feature):
    maxa = float(max(feature))
    return [(np.log(float(i)) / np.log(maxa)) for i in feature]

def get_feature(img_path):
    hu = get_hu(img_path)
    svd = get_svd(img_path)
    mr = get_mr(img_path)
    pzor = get_pzor(img_path)
    #sift = get_sift(img_path)
    minrectangle = get_minrectangle(img_path)
    #harris = get_harris(img_path)
    fourier = get_fourier(img_path)

    feature = np.concatenate(( mr, hu , pzor, minrectangle,svd,fourier))

    #feature = np.concatenate(( mr, hu))
    return feature

#
# def loadCsv(filename):
#     lines = csv.reader(open(filename, "r"))
#     dataset = list(lines)
#     for i in range(len(dataset)):
#         dataset[i] = [float(x) for x in dataset[i]]
#     return dataset
#
#
# def splitDataset(dataset, splitRatio):
#     trainSize = int(len(dataset) * splitRatio)
#     trainSet = []
#     copy = list(dataset)
#     while len(trainSet) < trainSize:
#         index = random.randrange(len(copy))
#         trainSet.append(copy.pop(index))
#     return [trainSet, copy]

def prepare_data(dir):
    features = []
    for i,img_dir in enumerate(sorted(os.listdir(dir))):
        img_folder = os.path.join(dir , img_dir)
        for item in os.listdir(img_folder):
            img_path = os.path.join(img_folder , item)
            feature = get_feature(img_path)
            feature = feature.tolist()
            feature.append(float(i+1))
            features.append(feature)
    return features


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    #print(separated)
    return separated


def mean(numbers):
    return sum([float(x) for x in numbers]) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(float(x) - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    print(summaries)
    return summaries


def calculateProbability(x, mean, stdev):
    if stdev == 0:
        exponent =  1
    else:
        exponent = math.exp(-(math.pow(float(x) - mean, 2) / (2 * math.pow(stdev, 2))))
    #print("stdev:",stdev)
    #print("exponent:",exponent)

    if stdev == 0:
        pro = 0.5
    else:
        pro = (1.0 / float((math.sqrt(2 * math.pi) * stdev))) * exponent
    #print("pro:", pro)
    return pro


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    print(correct)
    print(len(testSet))
    return (correct / float(len(testSet))) * 100.0


def main():
    # filename = 'pima-indians-diabetes.csv'
    # splitRatio = 0.67
    # dataset = loadCsv(filename)
    # #print(dataset)
    # trainingSet, testSet = splitDataset(dataset, splitRatio)
    # print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # print(trainingSet)
    # print(testSet)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    args = parser.parse_args()

    t11 = time.time()
    trainingSet = prepare_data(args.train_dir)
    data = np.array(trainingSet, dtype=np.float64)
    print(data.shape)
    # mean = np.mean(data, axis=0).astype(np.float64)
    # stdev = np.std(data, axis=0).astype(np.float64)
    # print(mean)
    # print(stdev)
    # data = (data - mean) / stdev
    train_data = data.tolist()
    testSet = prepare_data(args.test_dir)
    test_data = np.array(testSet, dtype=np.float64)
    # test_data = (test_data - mean) / stdev
    test_data = test_data.tolist()
    print("preparedata:",time.time()-t11)
    # prepare model
    t1 = time.time()
    summaries = summarizeByClass(train_data)
    print("t1:",time.time()-t1)
    # test model
    t2 = time.time()
    predictions = getPredictions(summaries, test_data)
    print("t2",time.time()-t2)
    print(predictions)
    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy: {0}%'.format(accuracy))

if __name__ == '__main__':
    main()
