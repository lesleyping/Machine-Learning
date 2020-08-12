#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:14:26
#
######
import argparse
import time
from lib.svm import svm_problem, svm_parameter
from lib.svmutil import *

def train(args):
    """
    使用图像的特征文件 来训练生成model文件
    :return:
    """
    y, x = svm_read_problem(args.train_file)
    param = svm_parameter('-s 0 -t 1 -c 5 -b 1')
    prob = svm_problem(y, x)
    model = svm_train(prob, param)
    svm_save_model(args.model_path, model)

def decode(args):
    yt, xt = svm_read_problem(args.test_file)
    model = svm_load_model(args.model_path)
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    cnt = 0
    for item in p_label:
        print('%d' % item, end =',')
        cnt += 1
        if cnt % 8 == 0:
            print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file")
    parser.add_argument("--model_path")
    parser.add_argument("--test_file")
    parser.add_argument("--decode", default= True)
    #parser.add_argument("--decode", default=False)
    args = parser.parse_args()
    if args.decode:
        print("decoding...")
        t2 = time.time()
        decode(args)
        print("t2:",time.time()-t2)
    else:
        print("training...")
        t1 = time.time()
        train(args)
        print("t1:",time.time()-t1)
