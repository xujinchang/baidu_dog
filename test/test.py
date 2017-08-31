#coding:utf-8
import numpy as np
import time
import os
import json
import sys
import socket
import logging
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import caffe
import random
import copy
import pickle

caffe.set_mode_gpu()
caffe.set_device(2)

MODEL_DEF = "model/deploy_resnext_fc.prototxt"
MODEL_PATH = "model/resnext_iter_4000.caffemodel"
MODEL_DEF1 = "model/deploy_l_resnext.prototxt"
MODEL_PATH1 = "model/l_resnext_iter_6000.caffemodel"
MODEL_DEF1_1 = "model/deploy_ll_resnext.prototxt"
MODEL_PATH1_1 = "model/ll_resnext_iter_6000.caffemodel"

MODEL_DEF3 = "model/deploy_resnext101.prototxt"
MODEL_PATH3 =  "model/resnext101_iter_20000.caffemodel"
MODEL_DEF3_1 = "model/deploy_l_resnext101.prototxt"
MODEL_PATH3_1 = "model/l_resnext101_iter_10000.caffemodel"

MODEL_DEF7 = "model/deploy_l_dpn92.prototxt"
MODEL_PATH7 = "model/l_dpn92_iter_14000.caffemodel"

MODEL_DEF8 = "model/deploy_resnet269_v2.prototxt"
MODEL_PATH8 = "model/resnet269_v2_iter_7000.caffemodel"


MODEL_DEF12 = "model/deploy_inception_resnet_v2.prototxt"
MODEL_PATH12 = "model/inception_resnet_v2_iter_3000.caffemodel"

MODEL_DEF13 = "model/deploy_inception_v3.prototxt"
MODEL_PATH13 = "model/v3_iter_6000.caffemodel"

MODEL_DEF14 = "model/deploy_inception_v4.prototxt"
MODEL_PATH14 = "model/v4_iter_4000.caffemodel"


mean = np.array((128., 128., 128.), dtype=np.float32)

def predict(image,the_net,SIZE,scale):
    inputs = []
    try:
        tmp_input = image
        tmp_input = cv2.resize(tmp_input,(SIZE,SIZE))
        tmp_input = np.subtract(tmp_input,mean)
        tmp_input = tmp_input.transpose((2, 0, 1))
        tmp_input = np.require(tmp_input, dtype=np.float32) * scale
    except Exception as e:
        raise Exception("Image damaged or illegal file format")
        return
    the_net.blobs['data'].reshape(1, *tmp_input.shape) 
    the_net.reshape()
    the_net.blobs['data'].data[...] = tmp_input
    the_net.forward()
    scores = the_net.blobs['prob'].data[0]
    return copy.deepcopy(scores)

if __name__=="__main__":  
    f = open("../data/test2.txt","rb")
    net = caffe.Net(MODEL_DEF, MODEL_PATH, caffe.TEST)
    net1 = caffe.Net(MODEL_DEF1, MODEL_PATH1, caffe.TEST)
    net1_1 = caffe.Net(MODEL_DEF1_1, MODEL_PATH1_1, caffe.TEST)
    net3 = caffe.Net(MODEL_DEF3, MODEL_PATH3, caffe.TEST)
    net3_1 = caffe.Net(MODEL_DEF3_1, MODEL_PATH3_1, caffe.TEST)
    net7 = caffe.Net(MODEL_DEF7, MODEL_PATH7, caffe.TEST)
    net8 = caffe.Net(MODEL_DEF8, MODEL_PATH8, caffe.TEST)
    net12 = caffe.Net(MODEL_DEF12, MODEL_PATH12, caffe.TEST)   
    net13 = caffe.Net(MODEL_DEF13, MODEL_PATH13, caffe.TEST)
    net14 = caffe.Net(MODEL_DEF14, MODEL_PATH14, caffe.TEST)

    dump1 = "predict/resnext50_2/"
    dump3 = "predict/resnext101/"
    dump7 = "predict/dpn92/"
    dump8 = "predict/resnet269_v2/"
    dump12 = "predict/inception_resnet_2/"
    dump13 = "predict/inception_v3_2/"
    dump14 = "predict/inception_v4_2/"
    imgs = list()
    for line in f.readlines():
        line = line.strip().split(" ")
        imgs.append(line[0])
    count = 0

    for img in imgs:
        count += 1
        if count==1:
            start_time = time.time()

        cv_img = cv2.imread(img)
        cv_img_flip = cv2.flip(cv_img, 1)
        scores1 = predict(cv_img,net,256,1)
        scores1_flip = predict(cv_img_flip,net,256,1)
        scores1_1 = predict(cv_img,net1,320,1)
        scores1_1_flip = predict(cv_img_flip,net1,320,1)
        scores1_1_1 = predict(cv_img,net1_1,384,1)
        scores1_1_1_flip = predict(cv_img_flip,net1_1,384,1)
        scores3 = predict(cv_img,net3,256,1)
        scores3_flip = predict(cv_img_flip,net3,256,1)
        scores3_1 = predict(cv_img,net3_1,320,1)
        scores3_flip_1 = predict(cv_img_flip,net3_1,320,1)

        scores7 = predict(cv_img,net7,320,1)
        scores7_flip = predict(cv_img_flip,net7,320,1)
        scores8 = predict(cv_img,net8,256,1)
        scores8_flip = predict(cv_img_flip,net8,256,1)

        scores12 = predict(cv_img,net12,331,0.0078125)
        scores12_flip = predict(cv_img_flip,net12,331,0.0078125)

        scores13 = predict(cv_img,net13,395,0.0078125)
        scores13_flip = predict(cv_img_flip,net13,395,0.0078125)

        scores14 = predict(cv_img,net14,299,0.0078125)
        scores14_flip = predict(cv_img_flip,net14,299,0.0078125)

        pickle.dump((scores1+scores1_flip+scores1_1+scores1_1_flip+scores1_1_1+scores1_1_1_flip)/6, open( dump1+img.split("/")[-1], "w"))
        pickle.dump((scores3+scores3_flip+scores3_1+scores3_flip_1)/4, open( dump3+img.split("/")[-1], "w"))
        pickle.dump((scores7+scores7_flip)/2, open( dump7+img.split("/")[-1], "w"))
        pickle.dump((scores8+scores8_flip)/2, open( dump8+img.split("/")[-1], "w"))
        pickle.dump((scores12+scores12_flip)/2, open( dump12+img.split("/")[-1], "w"))
        pickle.dump((scores13+scores13_flip)/2, open( dump13+img.split("/")[-1], "w"))
        pickle.dump((scores14+scores14_flip)/2, open( dump14+img.split("/")[-1], "w"))

        print "count: ",count
    f.close()
    end_time = time.time()
    run_time = end_time - start_time
    print "run_time: ",run_time
    print "per_run_time: ",float(run_time)/count