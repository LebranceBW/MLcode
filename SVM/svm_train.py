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
    (ax1, ax2) = plt.subplots(2, 1)[1]
    ax1.set_title(u'Liner SVM(Top),Gaussian SVM(Bottom)')
    ax1.set_ylabel(u"Sugar Rate")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.7)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax1.plot(next(pose_points), next(pose_points),'go')
    ax1.plot(next(counter_points), next(counter_points), 'bo')#样本点
    ax2.set_xlabel(u"Density")
    ax2.set_ylabel(u"Sugar Rate")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.7)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax2.plot(next(pose_points), next(pose_points),'go')
    ax2.plot(next(counter_points), next(counter_points), 'bo')#样本点
    return ax1, ax2

def decision_gaussian(sv, coef, gamma, x, bias):
    unitfunc = lambda coefunit, sv: coefunit[0] * np.exp(-gamma* np.sum(np.square(np.array(sv) - np.array(x))))
    return sum(map(unitfunc, coef, sv)) + bias
def main():
    '''
        主函数
    '''
    # 载入数据集
    assert os.path.exists(DATA_PATH), r"数据集不存在"
    y, x = svmutil.svm_read_problem(DATA_PATH)
    # 线性核SVM
    liner_model = svmutil.svm_train(y, x, '-s 0 -t 0 -c 100')
    wm_sv, wm_coef = liner_model.get_SV(), liner_model.get_sv_coef()
    omega = [0, 0]
    for sv, coef in itertools.zip_longest(wm_sv, wm_coef):
        omega[0] += sv[1] * coef[0] 
        omega[1] += sv[2] * coef[0] 
    bias = liner_model.get_sv_rho()
    ax1, ax2=plot_init()
    ax1.plot([-bias/omega[0],(omega[1]+bias)/(-omega[0])],[0,1])

    # 高斯核SVM
    gaussian_model = svmutil.svm_train(y, x, '-s 0 -t 2 -c 1000')
    wm_sv, wm_coef = gaussian_model.get_SV(), gaussian_model.get_sv_coef()
    sv = [[x[1], x[2]] for x in wm_sv]
    bias = gaussian_model.get_sv_rho()
    gamma = 0.5 #查看libsvm中，默认的γ为类别的倒数
    zeropoints = [[],[]]
    for x1 in [0.002*x for x in range(500)]:
        for x2 in [0.007*x for x in range(100)]:
            if np.abs(decision_gaussian(sv, wm_coef, gamma, [x1, x2], bias)) < 0.05:
                zeropoints[0].append(x1)
                zeropoints[1].append(x2)
    ax2.plot(zeropoints[0], zeropoints[1], '.r')
    plt.show()
if __name__ == "__main__":
    main()