#encoding:utf-8
'''
    使用线性SVM与高斯核的SVM分类西瓜数据集3.0α
'''
import os
import sys
from os import path
sys.path.append( path.dirname(  path.dirname(path.abspath(__file__) ) )) 
from libsvm import svmutil,svm
from dataSet.watermelon_3alpha import *
import matplotlib.pyplot as plt

def plot_Init():
    ax = plt.subplots()[1]
    fig2 = plt.gcf()
    fig2.canvas.set_window_title('Liner SVM')
    plt.title(u'Liner SVM')
    plt.xlabel(u"Density")
    plt.ylabel(u"Sugar Rate")
    plt.xlim(0, 1)
    plt.ylim(-0.2,0.5)
    return ax

def main():
    '''
        主函数
    '''
    dataPath = r".\dataSet\watermelon3.0alpha.libsvm"
    assert os.path.exists(dataPath), r"数据集不存在"
    y, x = svmutil.svm_read_problem(dataPath)
    model = svmutil.svm_train(y, x, '-t 0')
    wm_sv,wm_coef = model.get_SV(),model.get_sv_coef()
    omega = [0,0]
    for var in wm_sv:
        omega[0] += var[1]
        omega[1] += var[2]
    bias = 1 - wm_sv[0][1] * omega[0] - wm_sv[0][2] * omega[1]

    ax = plot_Init()
    for item in watermelon_posiexam_x:
        ax.plot(item[0], item[1], 'go')
    for item in watermelon_counterexample_x:
        ax.plot(item[0], item[1], 'bo')#样本点
    ax.plot([0,-bias/omega[0]],[-bias/omega[1],0])
    plt.show()
if __name__ == "__main__":
    main()