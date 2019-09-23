#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:52:32 2018

@author: Administrator
"""
import os
import caffe
import sys
import numpy as np
from sklearn import metrics
class predict_wheat(object):
    def __init__(self):
       self.deploy = 'C:/Profiles/Huang-project/TestModel/ts1mynet.prototxt'
       self.caffemodel = 'C:/Profiles/Huang-project/TestModel/rgb/rgb_ts1_iter_3000.caffemodel'
       self.imagedir =  'C:/Profiles/Huang-project/TestModel/rgb/val/'
       self.ground_truth_txt = 'C:/Profiles/Huang-project/TestModel/rgb/val.txt'
       self.label_file_txt ='C:/Profiles/Huang-project/TestModel/6.txt'
       self.meanfile = 'C:/Profiles/Huang-project/TestModel/rgb/mean.npy'
       self.predict_result_txt = "C:/Profiles/Huang-project/TestModel/rgb/res.txt"

    def Test_all(self):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(self.deploy,self.caffemodel,caffe.TEST)
        labels = np.loadtxt(self.label_file_txt, str, delimiter='\t')
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.meanfile))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))
        filelist=[]
        filenames = os.listdir(self.imagedir)
        for fn in filenames:
            fullfilename = os.path.join(self.imagedir,fn)
            filelist.append(fullfilename)
        for i in range(0, len(filelist)):
            print(i)
            img= filelist[i]
            im=caffe.io.load_image(img)
            net.blobs['data'].data[...] = transformer.preprocess('data',im)
            net.forward()
            prob= net.blobs['prob'].data.flatten()
            order=prob.argsort()[5]
            f=file(self.predict_result_txt,"a+")
            f.writelines(img+' '+labels[order]+'\n')

if __name__ == '__main__':
    predict_wheat1 = predict_wheat()
    predict_wheat1.Test_all()
