#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# Copyright 205@NPU. All Rights Reserved
#
# Licensed under the Apache License, Veresion 2.0(the "License");
# You may not use the file except in compliance with the Licese.
# You may obtain a copy of the License at
#
#   http://www.apache.org/license/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author lixping
#        congjian
# Date 2019/05/12 15:30:10
#
######################################################################


import os

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from feature import *

def get_feature(work_dir):
    hu = get_hu(work_dir)
    svd = get_svd(work_dir)
    mr = get_mr(work_dir)
    pzor = get_pzor(work_dir)
    #sift = get_sift(work_dir)
    minrectangle = get_minrectangle(work_dir)
    #harris = get_harris(work_dir)
    #fourier = get_fourier(work_dir)
    feature = np.concatenate(( mr, pzor,minrectangle,svd,hu))
    #feature1 = maxmin_normalization(feature)
    #feature2 = standard_normalization(feature)
    #feature3 = log_normalization(feature)
    #feature = np.concatenate(( mr, hu))
    #return feature
    return feature

def lda(x, y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x, y)
    res = lda.transform(x)
    return res

def pca(x):
    pca = PCA(n_components=2)
    res = pca.fit_transform(x)
    return res

def t_sne(x):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne.fit_transform(x)
    res = tsne.embedding_
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="5f2d/train")
    parser.add_argument("--res_dir", default="data_visual_res_jcong")
    args = parser.parse_args()

    x = []
    y = []
    for label, class_dir in tqdm(enumerate(sorted(os.listdir(args.work_dir)))):
        for img in tqdm(sorted(os.listdir(os.path.join(args.work_dir, class_dir)))):
            work_dir = os.path.join(args.work_dir, class_dir, img)
            # feature = get_feature(work_dir)
            # x.append(feature)
            y.append(label)
    x = np.array(x).astype(np.float64)
    y = np.array(y).astype(np.int64).transpose()
    np.savetxt("y.txt",y)

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)



    # plot pca
    # res = pca(x)
    # np.savetxt("1.txt",res)
    res = np.loadtxt("1.txt")
    plt.figure(figsize = (8, 5))
    plt.scatter(res[:, 0], res[:, 1], c = y, marker='o')
    plt.title("PCA")
    plt.savefig(os.path.join(args.res_dir,"pca.png"))

    # plot t-sne
    # res = t_sne(x)
    # np.savetxt("2.txt", res)
    res = np.loadtxt("2.txt")
    plt.figure(figsize = (8, 5))
    plt.scatter(res[:, 0], res[:, 1], c = y, marker = 'o')
    plt.title("T-SNE")
    plt.savefig(os.path.join(args.res_dir, "t-sne.png"))

    # plot lda
    # res = lda(x, y)
    # np.savetxt("3.txt", res)
    res = np.loadtxt("2.txt")
    plt.figure(figsize=(8, 5))
    plt.scatter(res[:, 0], res[:, 1], c = y, marker = 'o')
    plt.title("LDA")
    plt.savefig(os.path.join(args.res_dir, "lda.png"))
