#encoding:utf-8
'''
    将西瓜3.0α数据集转换为libsvm格式
'''
import sys
from os import path
sys.path.append(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
from dataSet.watermelon_3alpha import *

model = "{label} 1:{value1} 2:{value2}\n"
with open(r".\dataSet\watermelon3.0alpha.libsvm",'w') as f:
    stringlist = []
    for index in range(17):
        stringlist.append(model.format(label = ['-1','+1'][watermelon_y[index]], value1 = watermelon_x[index][0], 
        value2 = watermelon_x[index][1]))
    f.writelines(stringlist)