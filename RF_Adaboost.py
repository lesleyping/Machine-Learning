#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:14:26
#
######

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 单层决策树
class DecisionStump:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = self.X.size()[0]

    def train(self, W, steps=100):
        min_v = float("inf")
        threshold_value = 0
        threshold_pos = 0
        threshold_tag = 0
        self.W = torch.Tensor(W)
        for i in range(self.N):  # value表示阈值，errcnt表示错误的数量
            value, errcnt = self.find_min(i, 1, steps)
            if (errcnt < min_v):
                min_v = errcnt
                threshold_value = value
                threshold_pos = i
                threshold_tag = 1

        for i in range(self.N):  # -1
            value, errcnt = self.find_min(i, -1, steps)
            if (errcnt < min_v):
                min_v = errcnt
                threshold_value = value
                threshold_pos = i
                threshold_tag = -1
        self.threshold_value = threshold_value
        self.threshold_pos = threshold_pos
        self.threshold_res = threshold_tag
        print(self.threshold_value, self.threshold_pos, self.threshold_res)
        return min_v

    def find_min(self, i, tag, steps):
        t = 0
        tmp = self.predintrain(self.X, i, t, tag).transpose(0, 1)
        # print(type(tmp))
        # print(type(self.y))
        # ttt = tmp != self.y
        # print("====", (tmp.cpu() != self.y.cpu()).size())
        # print(self.W.size())
        # errcnt = torch.sum((tmp != self.y).float() * self.W)
        # print now
        buttom = torch.min(self.X[i, :])  # 该项属性的最小值，下界
        up = torch.max(self.X[i, :])  # 该项属性的最大值，上界
        minerr = float("inf")  # 将minerr初始化为无穷大
        value = 0  # value表示阈值
        st = (up - buttom) / steps  # 间隔
        if st!=0:
            for t in torch.arange(buttom, up, st):
                tmp = self.predintrain(self.X, i, t, tag).transpose(0, 1)
                tmp = tmp.float()
                errcnt = torch.sum((tmp != self.y).float() * self.W)
                if errcnt < minerr:
                    minerr = errcnt
                    value = t
        return value, minerr

    def predintrain(self, test_set, i, t, tag):  # 训练时按照阈值为t时预测结果
        test_set = test_set.view(self.N, -1)
        pre_y = torch.ones((test_set.size()[1], 1))

        pre_y[test_set[i, :] * tag < t * tag] = -1

        return pre_y

    def pred(self, test_X):  # 弱分类器的预测
        test_X = torch.Tensor(test_X).view(self.N, -1)  # 转换为N行X列，-1懒得算
        pre_y = torch.ones((torch.Tensor(test_X).size()[1], 1))
        pre_y[test_X[self.threshold_pos, :] * self.threshold_res < self.threshold_value * self.threshold_res] = -1
        return pre_y


class AdaBoost:
    def __init__(self, X, y, Weaker=DecisionStump):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y).flatten()
        self.Weaker = Weaker

        self.sums = torch.zeros(self.y.shape)
        '''
            W为权值，初试情况为均匀分布，即所有样本都为1/n
        '''
        self.W = torch.ones((self.X.size()[1], 1)).flatten() / self.X.size()[1]

        self.Q = 0  # 弱分类器的实际个数

    # M 为弱分类器的最大数量，可以在main函数中修改
    def train(self, M=5):
        self.G = {}  # 表示弱分类器的字典
        self.alpha = {}  # 每个弱分类器的参数
        for i in range(M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(M):  # self.G[i]为第i个弱分类器
            self.G[i] = self.Weaker(self.X, self.y)
            e = self.G[i].train(self.W)  # 根据当前权值进行该个弱分类器训练
            self.alpha[i] = 1.0 / 2 * torch.log((1 - e) / e)  # 计算该分类器的系数
            res = self.G[i].pred(self.X)  # res表示该分类器得出的输出

            # 计算当前次数训练精确度
            print("weak classfier acc", accuracy_score(self.y,
                                                       res), "\n======================================================")

            # Z表示规范化因子
            Z = self.W * torch.exp(-self.alpha[i] * self.y * res.transpose(1, 0))
            self.W = (Z / Z.sum()).flatten()  # 更新权值
            self.Q = i
            # errorcnt返回分错的点的数量，为0则表示perfect
            if (self.errorcnt(i) == 0):
                print("%d个弱分类器可以将错误率降到0" % (i + 1))
                break

    def errorcnt(self, t):  # 返回错误分类的点
        self.sums = self.sums + self.G[t].pred(self.X).flatten() * self.alpha[t]

        pre_y = torch.zeros_like(torch.Tensor(self.sums))
        pre_y[self.sums >= 0] = 1
        pre_y[self.sums < 0] = -1

        t = (pre_y != self.y).sum()
        return t

    def pred(self, test_X):  # 测试最终的分类器
        test_X = torch.Tensor(test_X)
        sums = torch.zeros(test_X.size()[1])
        for i in range(self.Q + 1):
            sums = sums + self.G[i].pred(test_X).flatten() * self.alpha[i]
        pre_y = torch.zeros_like(torch.Tensor(sums))
        pre_y[sums >= 0] = 1
        pre_y[sums < 0] = -1
        return pre_y


def main():
    # # load data
    # dataset = np.loadtxt('data.txt', delimiter=",")
    # x = dataset[:, 0:8]
    # y = dataset[:, 8]
    #
    # # prepare train data
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # load data
    train_dataset = np.loadtxt('p_c_train.txt', delimiter=",")
    test_dataset = np.loadtxt('p_c_test.txt', delimiter=",")
    print(len(train_dataset[0]))
    x_train = train_dataset[:, 0:261]
    y_train = train_dataset[:, 261]
    x_test = test_dataset[:, 0:261]
    y_test = test_dataset[:, 261]

    # prepare test and train data
    x_train = x_train.transpose()
    y_train[y_train == 1] = 1
    y_train[y_train == 2] = -1

    x_test = x_test.transpose()
    y_test[y_test == 1] = 1
    y_test[y_test == 2] = -1

    # train
    ada = AdaBoost(x_train, y_train)
    ada.train(50)

    # predict
    y_pred = ada.pred(x_test)
    y_pred = y_pred.numpy()
    print("total test", len(y_pred))
    print("true pred", len(y_pred[y_pred == y_test]))
    print("acc", accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()
