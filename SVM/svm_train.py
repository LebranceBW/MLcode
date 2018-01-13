#encoding:utf-8
'''
    使用线性SVM与高斯核的SVM分类西瓜数据集3.0α
'''
import os
import sys
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libsvm import svmutil
from dataSet.watermelon_3alpha import watermelon_counterexample_x, watermelon_posiexam_x
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r".\dataSet\watermelon3.0alpha.libsvm"
def plot_init():
    '''
    初始化matplot
    '''
    plt.subplot(211)
    plt.figure(1)
    plt.title(u'Liner SVM(Top),Gaussian SVM(Bottom)')
    plt.ylabel(u"Sugar Rate")
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.5)
    plt.subplot(212)
    plt.xlabel(u"Density")
    plt.ylabel(u"Sugar Rate")
    plt.xlim(0, 1)
    plt.ylim(-0.2, 0.5)

def main():
    '''
        主函数
    '''
    assert os.path.exists(DATA_PATH), r"数据集不存在"
    y, x = svmutil.svm_read_problem(DATA_PATH)
    liner_model = svmutil.svm_train(y, x, '-s 0 -t 0 -c 100')
    
    wm_sv, wm_coef = liner_model.get_SV(), liner_model.get_sv_coef()
    omega = [0, 0]
    for sv, coef in itertools.zip_longest(wm_sv, wm_coef):
        omega[0] += sv[1] * coef[0] 
        omega[1] += sv[2] * coef[0] 
    bias = liner_model.get_sv_rho()
    plot_init()
    plt.subplot(211)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    plt.plot(next(pose_points), next(pose_points),'go')
    plt.plot(next(counter_points), next(counter_points), 'bo')#样本点
    plt.plot([-bias/omega[0],(omega[1]+bias)/(-omega[0])],[0,1])
    plt.subplot(212)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    plt.plot(next(pose_points), next(pose_points),'go')
    plt.plot(next(counter_points), next(counter_points), 'bo')#样本点

    plt.show()
if __name__ == "__main__":
    main()