#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:19:34
#
######
import torch.utils.data as data
import os
import numpy as np
import torch
import cv2

class Infrared_Dataloader(data.Dataset):
    def __init__(self, training_file, test_file, label_dim, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.training_file = training_file
        self.test_file = test_file
        self.train = train  # training set or test set
        self.label_dim = 8
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data = open(data_file).readlines()

    def __getitem__(self, index):
        img_path, target = self.data[index].split(" ")
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (350, 250))
        img = np.array(img, dtype="float32") / 255
        #img = img[np.newaxis, :, :]
        if self.transform is not None:
            img = self.transform(img)
        # img = np.expand_dims(img, axis=2)# [338,328] [338,328,1]
        # return torch.from_numpy(img).long(), torch.tensor(int(target))
        return img, torch.tensor(int(target))

    def __len__(self):
        return len(self.data)
#CUDA_VISIBLE_DEVICES=2