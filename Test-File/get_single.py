#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import caffe
import numpy as np
from sklearn import metrics

ground_truth_txt = 'C:/Profiles/Huang-project/TestModel/rgb/val.txt'
predict_result_txt = "C:/Profiles/Huang-project/TestModel/rgb/res.txt"
imagedir =  'C:/Profiles/Huang-project/TestModel/rgb/val/'
filelist=[]
filenames = os.listdir(imagedir)
for fn in filenames:
    fullfilename = os.path.join(imagedir,fn)
    filelist.append(fullfilename)
# 计算混淆矩阵以及识别结果
ground_truth = []
pre = []
fp_gt = open(ground_truth_txt,'r')
fp_pr = open(predict_result_txt, 'r')
for i in range(0, len(filelist)):
    lines_gt = fp_gt.readline()
    ground_truth.append(lines_gt.split()[-1])
    lines_pr = fp_pr.readline()
    pre.append(lines_pr.split()[-1])
fp_gt.close()
fp_pr.close()
ClaReslut=metrics.confusion_matrix(ground_truth, pre)
print ("ClassResult:")
print (ClaReslut)
f = (float(ClaReslut[0][0])/sum(ClaReslut[0]))
a = (float(ClaReslut[1][1])/sum(ClaReslut[1]))
b = (float(ClaReslut[2][2])/sum(ClaReslut[2]))
c = (float(ClaReslut[3][3])/sum(ClaReslut[3]))
d = (float(ClaReslut[4][4])/sum(ClaReslut[4]))
e = (float(ClaReslut[5][5])/sum(ClaReslut[5]))
acc1 = 0.0
acc2 = 0.0
for i in range(6):
    acc1 += ClaReslut[i][i]
    acc2 += sum(ClaReslut[i])
accuracy = float(acc1) / acc2
recall = float(a+b+c+d+f+e) / 6.0
print ("recall:"), (float(recall))
print ("accuracy:"), (float(accuracy))
print ("normal:"), (float(ClaReslut[0][0])/sum(ClaReslut[0]))
print ("broken:"), (float(ClaReslut[1][1])/sum(ClaReslut[1]))
print ("worm:"), (float(ClaReslut[2][2])/sum(ClaReslut[2]))
print ("disease:"), (float(ClaReslut[3][3])/sum(ClaReslut[3]))
print ("bud:"), (float(ClaReslut[4][4])/sum(ClaReslut[4]))
print ("mold:"), (float(ClaReslut[5][5])/sum(ClaReslut[5]))
print ("=================================================")
