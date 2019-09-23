# Caffe
CNN Based On Caffe -- 对小麦的分类识别。  
运行环境：python2.7 + cuda9.1 + cudnn  
## 结构
- GenModel-训练模块
- GenPng-训练过程可视化
- TestModel-测试模块
- Result
## GenModel
训练模块，其中NAME2TXT.exe是将文件夹下的所有图片的名字提取出来存储在txt中。  
使用方法：将NAME2TXT.exe放在图片文件夹下，运行改软件，点击button即可。生成完后需要删除生成的txt文件最后两行。  
其他网络：resnet-18_1.caffemodel文件是修改了layer层的名字，但各层权重未变，当使用两层resnet18时，加载两个caffemodel即可，两个模型以逗号隔开。
## GenPng
训练过程可视化，将caffe生成的log文件可视化出来，效果如下  
![](https://raw.githubusercontent.com/Shuaikun-Huang/Caffe/master/GenPng/net3/net3.png  "net3")  
使用方法：cmd下执行  
``` python parse_log.py yourlogname.log .\ ```  
上面执行完后，会生成一个test和一个train文件，查看这两个文件，并修改gen_pic.py中对应的key值，让其对应。最后cmd下接着运行  
``` python gen_pic.py ```  
## TestModel
测试模块，即测试生成的模型。使用方法：  
1. 测试单个模型： 先运行test_single.py生成预测结果，在运行get_single.py生成预测矩阵信息。运行前需修改文件路径。
2. 测试多个模型： run_all.py。适用于多个网络与多个测试集，也需配置路径。
## Result
WFFN的Accuracy达到92%。
