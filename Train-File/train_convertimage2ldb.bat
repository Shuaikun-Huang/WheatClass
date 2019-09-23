SET GLOG_logtostderr=1
..\Release\convert_imageset.exe --backend=leveldb --shuffle .\img\train\ .\img\train.txt .\img\trainldb 1
pause
