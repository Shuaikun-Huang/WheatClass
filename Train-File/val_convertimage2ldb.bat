SET GLOG_logtostderr=1
..\Release\convert_imageset.exe --backend=leveldb --shuffle .\img\val\ .\img\val.txt .\img\valldb 1
pause
