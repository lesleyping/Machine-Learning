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

import argparse
from feature2 import *
from tqdm import tqdm
import time
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

    feature = np.concatenate(( mr, pzor, minrectangle, hu, svd,fourier))

    #feature1 = maxmin_normalization(feature)
    #feature2 = standard_normalization(feature)
    #feature3 = log_normalization(feature)
    #feature = np.concatenate(( mr, hu))
    #return feature
    return feature

def convert_value_to_str(label, feature):
    fmt = ''
    for item in feature:
        fmt = fmt + '{},'.format(item)
    line = fmt + '{}'.format(label)
    return  line

def imgs2feature(label, data_file, img_folder):
    for item in tqdm(os.listdir(img_folder)):
        img_path = os.path.join(img_folder , item)
        feature = get_feature(img_path)
        line = convert_value_to_str(label,feature)
        data_file.write(line + "\n")


def get_txt(in_dir,out_file):
    data_file = open(out_file, 'w')
    for i,img_dir in tqdm(enumerate(sorted(os.listdir(in_dir)))):
        img_folder = os.path.join(in_dir,img_dir)
        imgs2feature(i+1 , data_file , img_folder)
    data_file.close()

def svm_convert_value_to_str(label, feature):
   line = '{} '.format(label)
   for i, item in enumerate(feature):
       fmt = '{}:{} '.format(i + 1, item)
       line = line + fmt
   return line

def svm_imgs2feature_file(label, svm_feature_file, img_folder):
    for item in os.listdir(img_folder):
        img_path = os.path.join(img_folder, item)
        feature = get_feature(img_path)
        line = wvm_convert_value_to_str(label, feature)
        svm_feature_file.write(line + "\n")

def get_svm_txt(in_dir, out_file):
    svm_feature_file = open(out_file, 'w')
    for i, img_dir in enumerate(sorted(os.listdir(in_dir))):
        img_folder = os.path.join(in_dir,img_dir)
        svm_imgs2feature_file(i + 1, svm_feature_file, img_folder)
    svm_feature_file.close()

if __name__ == '__main__':
    '''
    process on RF_Adaboost
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    parser.add_argument("--data_train")
    parser.add_argument("--data_test")
    args = parser.parse_args()
    t1 = time.time()
    get_txt(args.train_dir,args.data_train)
    get_txt(args.test_dir,args.data_test)
    print("t:",time.time()-t1)
    '''
    Process on svm
    '''
    get_svm_txt(args.train_dir, args.data_train)
    get_svm_txt(args.test_dir, args.data_test)
    
