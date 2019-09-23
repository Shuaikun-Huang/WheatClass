#!/usr/bin/env python
# -*- coding: utf-8 -*-
#使用前请修改文件路径
import os
import caffe
import sys
import numpy as np
from sklearn import metrics

#测试集数量, 第一个文件夹是test0
test_num = 2;
#测试网络
deploy = 'D:/Profiles/Huang-project/WheatClass/Test-File/Wheatnet/Wheatnet_deploy.prototxt'
#模型地址
model_dir = 'D:/Profiles/Huang-project/WheatClass/Test-File/Wheatnet/snapshot/'
#分的类
label_file_txt ='D:/Profiles/Huang-project/WheatClass/Test-File/img/6.txt'
#mean file
meanfile = 'D:/Profiles/Huang-project/WheatClass/Test-File/img/mean.npy'

#获取目录下的model文件
files = []
for root,dirs,filess in os.walk(model_dir):
    files = filess
    break
#遍历每一个model
for j in range(len(files)):
    #获取model的路径
    caffemodel = model_dir + files[j]
    #测试5次,需要准备5个test集，分别为test1~5
    for i in range(0, test_num):
        #图片地址
        imagedir =  "D:/Profiles/Huang-project/WheatClass/Test-File/img/test%d"%i
        #真实分类txt文件
        ground_truth_txt = "D:/Profiles/Huang-project/WheatClass/Test-File/img/test%d.txt"%i
        #预测分类文件存储位置即名字
        predict_result_txt = "D:/Profiles/Huang-project/WheatClass/Test-File/Wheatnet/predict_result_txt/%s_%d.txt"%(files[j][:-11],i)
        #分类结果矩阵等文件地址及名字
        result =  "D:/Profiles/Huang-project/WheatClass/Test-File/Wheatnet/result/%s_%d.txt"%(files[j][:-11],i)
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(deploy, caffemodel, caffe.TEST)
        labels = np.loadtxt(label_file_txt, str, delimiter='\t')
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(meanfile))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))
        #读取测试集图片
        filelist=[]
        filenames = os.listdir(imagedir)
        for fn in filenames:
            fullfilename = os.path.join(imagedir,fn)
            filelist.append(fullfilename)

        for ii in range(0, len(filelist)):
            print(ii)
            img= filelist[ii]
            im=caffe.io.load_image(img)
            net.blobs['data'].data[...] = transformer.preprocess('data',im)
            net.forward()
            prob= net.blobs['prob'].data.flatten()
            order=prob.argsort()[5]
            f=file(predict_result_txt,"a+")
            f.writelines(img+' '+labels[order]+'\n')
        f.close()
        #第一个测试集写predict_txt完成


        # 计算混淆矩阵以及识别结果
        ground_truth = []
        pre = []
        fp_gt = open(ground_truth_txt,'r')
        fp_pr = open(predict_result_txt, 'r')
        for _ in range(len(filelist)):
            lines_gt = fp_gt.readline()
            ground_truth.append(lines_gt.split()[-1])
            lines_pr = fp_pr.readline()
            pre.append(lines_pr.split()[-1])
        fp_gt.close()
        fp_pr.close()
        ClaReslut=metrics.confusion_matrix(ground_truth, pre)
        f = (float(ClaReslut[0][0])/sum(ClaReslut[0]))
        a = (float(ClaReslut[1][1])/sum(ClaReslut[1]))
        b = (float(ClaReslut[2][2])/sum(ClaReslut[2]))
        c = (float(ClaReslut[3][3])/sum(ClaReslut[3]))
        d = (float(ClaReslut[4][4])/sum(ClaReslut[4]))
        e = (float(ClaReslut[5][5])/sum(ClaReslut[5]))
        acc1 = 0.0
        acc2 = 0.0
        for ii in range(6):
            acc1 += ClaReslut[ii][ii]
            acc2 += sum(ClaReslut[ii])
        accuracy = float(acc1) / acc2
        recall = float(a+b+c+d+f+e) / 6.0

        #写结果
        w = file(result,"a+")
        w.writelines("Class Result: " + "\n")
        w.write(str(ClaReslut) + '\n')
        w.writelines("recall:  " + str(recall) + "\n")
        w.writelines("accuracy:   " + str(accuracy) + "\n")
        w.writelines("normal:   " + str(float(ClaReslut[0][0])/sum(ClaReslut[0])) + "\n")
        w.writelines("broken:   " + str(float(ClaReslut[1][1])/sum(ClaReslut[1])) + "\n")
        w.writelines("worm:   " + str(float(ClaReslut[2][2])/sum(ClaReslut[2])) + "\n")
        w.writelines("disease:   " + str(float(ClaReslut[3][3])/sum(ClaReslut[3])) + "\n")
        w.writelines("bud:   " + str(float(ClaReslut[4][4])/sum(ClaReslut[4])) + "\n")
        w.writelines("mold:   " + str(float(ClaReslut[5][5])/sum(ClaReslut[5])) + "\n")
        w.close()
        print("test%d finshed"%i)
    print("%s finshed"%files[j])
    print("====================================================")
print("Done")
