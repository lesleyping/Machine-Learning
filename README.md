# Machine-Learning
Simple image classification using various machine learning methods, including image feature extraction steps and feature dimensionality reduction visualization method.
> 用多种机器学习方法进行简单的图像分类（针对灰度图像，也可改为可见光图像），包括对图像的特征提取步骤及特征降维可视化方法。

## 1.Prepare Data
_涉及到的文件包括：prapare_data.py, prepare_data.sh, feature2.py_

1.1 数据采用如下格式存放

* |-- data
    * |--train
        * |--1 
        * |--2 
        * |-... 
    * |--test
        * |--1 
        * |--2 
        * |-...

1.2 运行说明

**RF、Adaboost、SVM**方法的数据准备使用prapare_data.py与prepare_data.sh两个文件进行准备; 

**knn、Bayes**方法不需要数据准备，直接运行即可.

1.3 特征
> 图像的特征包括
>>SVD特征\
>>Hu不变矩特征\
>>均值对比度特征（MR）\
>>亮度比(Pzor):比目标最亮点亮度小10%以内的像素点个数与目标总像素个数之间的比值\
>>长宽比(minrectangle):目标最小外接矩形的长度与宽度之比值\
>>SIFT特征\
>>Harris特征\
>>Fourier傅立叶描述子



## 2.Methods
_涉及到的文件包括：RF_Adaboost.py,RF_Adaboost-multi_classes.py, SVM.py, 
SVM.sh, knn.py, knn.sh, cnn/*_

### 2.1 AdaBoost
RF_Adaboost.py和RF_Adaboost-multi_classes.py 分别针对步骤1提取到的特征，实现了图像的二分类与多分类。

### 2.2 Random Forest
RF_Adaboost.py和RF_Adaboost-multi_classes.py 分别针对步骤1提取到的特征，实现了图像的二分类与多分类。

### 2.3 SVM
SVM.py\
SVM.sh

### 2.4 KNN
knn.py\
knn.sh

### 2.5 CNN
使用简单的神经网络进行图像分类

cnn/*


## 3.feature dimensionality reduction &visualization


data_visualization.py

实现了数据的三种降维方法

包括常见的LDA、PCA、T_SNE三种方法

并对结果进行可视化与保存，可以显示类别间的差异，验证特征的有效性与可分性

结果展示：
![lda]()