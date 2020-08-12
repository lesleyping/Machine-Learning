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
import numpy as np
import cv2

def calculate_mean_stdev(train_dir):
    imgs = np.zeros([328,338,1])
    for i, img_dir in enumerate(sorted(os.listdir(train_dir))):
        img_folder = os.path.join(train_dir,img_dir)
        for item in os.listdir(img_folder):
            img_path = os.path.join(img_folder, item)
            img = cv2.imread(img_path,0)
            img = img[:,:,np.newaxis]
            imgs = np.concatenate((imgs,img), axis=2)
    imgs = imgs.astype(np.float32) /255
    pixels = imgs.ravel()
    means = np.mean(pixels)
    stdev = np.std(pixels)
    print("means:{}".format(means))
    print("stdev:{}".format(stdev))




def process_data(in_dir, out_file):
    out_file = open(out_file, 'w')
    for i, img_dir in enumerate(sorted(os.listdir(in_dir))):
        img_folder = os.path.join(in_dir,img_dir)
        for item in os.listdir(img_folder):
            line = os.path.abspath(os.path.join(img_folder, item)) + " " + str(i) + "\n"
            out_file.write(line)
    out_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    parser.add_argument("--train_file")
    parser.add_argument("--test_file")
    args = parser.parse_args()
    process_data(args.train_dir, args.train_file)
    process_data(args.test_dir, args.test_file)
    #calculate_mean_stdev(args.train_dir)
