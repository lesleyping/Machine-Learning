#!/usr/bin/env python
# -*- coding: utf-8 -*-

######
#
# Mail   npuxpli@mail.nwpu.edu.cn
# Author LiXiping
# Date 2019/09/20 16:19:34
#
######

import cv2
import math
import numpy as np
from numpy import linalg as la
from skimage import measure
from  PIL import Image

def get_svd(img_path):
    img = cv2.imread(img_path, 0)
    U,sigma,VT=la.svd(img)
    # print(sigma)
    #print("svd sigma")
    #print(type(sigma))
    # sigma[::-1].sort()
    sigma = sigma[:10]
    return sigma

def get_hu(img_path):
    img = cv2.imread(img_path, 0)
    moments = cv2.moments(img)
    humoments = cv2.HuMoments(moments)
    # humoments = np.log(np.abs(humoments))
    humoments = humoments.transpose()[0]
    hu1 = np.log(np.abs(humoments[0]))
    hu2 = np.log(np.abs(humoments[1]))
    hu = '{},{}'.format(hu1, hu2)
    hu = hu.strip(',').split(',')
    #print(hu)
    return hu


def get_mr(img_path):
    def im2double(im):
        min_val = np.min(im)
        max_val = np.max(im)
        #out = (im.astype(np.float32) - min_val) / (max_val - min_val)
        out = im.astype(np.float32)
        return out

    image_in = cv2.imread(img_path, 0)
    # Otsu滤波
    threshold, image_BW = cv2.threshold(image_in ,0,255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    orign_image=im2double(np.array(image_in))
    binary_image=im2double(np.array(image_BW))
    [m, n] = image_in.shape
    sum_foreground=0
    sum_background=0
    count_foreground=0
    count_background=0

    for row in range(m):
        for col in range(n):
            if binary_image[row,col] != 0:
                sum_foreground = sum_foreground + orign_image[row,col]
                count_foreground = count_foreground + 1
            else:
                sum_background = sum_background + orign_image[row,col]
                count_background = count_background + 1

    m1 = sum_foreground / count_foreground
    m2 = sum_background / count_background
    if m2 != 0:
        MR = m1 / m2
    else:
        MR = 0

    return np.array([MR])

def get_pzor(img_path):
    def im2double(im):
        min_val = np.min(im)
        max_val = np.max(im)
        out = (im.astype(np.float32) - min_val) / (max_val - min_val)
        return out

    image_in = cv2.imread(img_path, 0)
    threshold, image_BW = cv2.threshold(image_in ,0,255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    orign_image=im2double(np.array(image_in))
    binary_image=im2double(np.array(image_BW))

    [m,n] = image_in.shape

    count = 0
    light = 0
    B = np.ones((m,n))*0

    for row in range(m):
        for col in range(n):
            if binary_image[row , col] != 0:
                B[row, col] = orign_image[row,col]
                count = count + 1
    thresh =(np.max(B))*0.9
    for row in range(m):
        for col in range(n):
            if orign_image[row , col] > thresh:
                light = light + 1
    light = float(light)
    count = float(count)
    # print(light)
    # print(count)
    Pzor = float(light/count)
    return np.array([Pzor])

def get_minrectangle(img_path):
    image_in = cv2.imread(img_path, 0)
    threshold, image_BW = cv2.threshold(image_in ,0,255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    labels = measure.label(image_BW, connectivity=2)
    regions = measure.regionprops(labels)

    area_max = 0
    for r in regions:
        if r.area > area_max:
            area_max = r.area
            x = float(r.bbox[0])
            y = float(r.bbox[1])
            width = float(r.bbox[3] - r.bbox[1])
            height = float(r.bbox[2] - r.bbox[0])
            ration = float(height /width)
            extent = r.extent

    minrectangle = '{},{},{},{},{},{},{}'.format(x, y, width, height, area_max, ration, extent)
    minrectangle = minrectangle.strip(',').split(',')
    #print(minrectangle)
    return minrectangle

def get_sift(img_path):
    img = cv2.imread(img_path , 0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img , None)
    num = len(kp)

    return np.array([num])

def get_harris(img_path):
    img = cv2.imread(img_path , 0)
    img = np.float32(img)
    dst = cv2.cornerHarris(img ,2 ,3,0.04)
    dst = cv2.dilate(dst , None)
    num = len(dst)
    return np.array([num])


def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""
    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    descriptors = descriptors[
        center_index - degree / 2:center_index + degree / 2]
    descriptors = np.fft.ifftshift(descriptors)
    return descriptors

def reconstruct(descriptors, degree):
    """ reconstruct(descriptors, degree) attempts to reconstruct the image
    using the first [degree] descriptors of descriptors"""
    # truncate the long list of descriptors to certain length
    descriptor_in_use = truncate_descriptor(descriptors, degree)
    contour_reconstruct = np.fft.ifft(descriptor_in_use)
    contour_reconstruct = np.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    # make positive
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    # normalization
    contour_reconstruct *= 800 / contour_reconstruct.max()
    # type cast to int32
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)
    black = np.zeros((800, 800), np.uint8)
    # draw and visualize
    # cv2.drawContours(black, contour_reconstruct, -1, 255, thickness=-1)
    # cv2.imshow("black", black)
    # cv2.waitKey(1000)
    # cv2.imwrite("reconstruct_result.jpg", black)
    # cv2.destroyAllWindows()
    return descriptor_in_use

# def addNoise(descriptors):
#     """this function adds gaussian noise to descriptors
#     descriptors should be a [N,2] numpy array"""
#     scale = descriptors.max() / 10
#     noise = np.random.normal(0, scale, descriptors.shape[0])
#     noise = noise + 1j * noise
#     descriptors += noise


def get_fourier(img_path):
    img = cv2.imread(img_path , 0)
    retval, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contour = []
    _, contour, hierarchy = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
        contour
    )
    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[: , 0]
    contour_complex.imag = contour_array[: , 1]
    fourier_result = np.fft.fft(contour_complex)

    contour = reconstruct(fourier_result, 2)
    #addNoise(contour)
    contour = np.absolute(contour)

    return contour




# path = '063.bmp'
# # a = Image.open(path)
# # a = np.array(a)
# s = get_svd(path)
# print(s)


# a,b,c = get_svd(path)
# a.shape = (328,1)
# # a = np.transpose(a)
# print(a.shape)
# d = b[:,:]*a[50:]*c[:50]

# print(a)
# print(type(contour))
# print(contour)


