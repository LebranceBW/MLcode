#encoding:utf-8
'''
    使用线性SVM与高斯核的SVM分类西瓜数据集3.0α
'''
import os
import sys
import itertools
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(basedir)
from lib.libsvm import svmutil
from dataSet.watermelon_3alpha import watermelon_counterexample_x, watermelon_posiexam_x
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from functools import partial, reduce

DATA_PATH = basedir + r"/dataSet/watermelon3.0alpha.libsvm"
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
    ax1.plot(next(pose_points), next(pose_points), 'go')
    ax1.plot(next(counter_points), next(counter_points), 'bo')#样本点
    ax2.set_xlabel(u"Density")
    ax2.set_ylabel(u"Sugar Rate")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.7)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax2.plot(next(pose_points), next(pose_points), 'go')
    ax2.plot(next(counter_points), next(counter_points), 'bo')#样本点
    return ax1, ax2

def decision_gaussian(svs, coef, gamma, bias):
    '''
        根据高斯函数的参数计算结果
        输入 支持向量， 权值， gamma， 偏差
        返回一个决策函数
    '''
    def func(feature_vector):
        unitfunc = lambda coefunit, sv: coefunit[0] * np.exp(-gamma* np.sum(np.square(np.array(sv) - np.array(feature_vector)), axis=1))
        return sum(map(unitfunc, coef, svs)) + np.array(bias)
    return func

def main():
    '''
        主函数
    '''
    # 载入数据集
    assert os.path.exists(DATA_PATH), r"数据集不存在"
    labels, feature_spaces = svmutil.svm_read_problem(DATA_PATH)
    #1 线性核SVM
    liner_model = svmutil.svm_train(labels, feature_spaces, '-h 0 -t 0 -c 100 -q')
    wm_sv, wm_coef = liner_model.get_SV(), liner_model.get_sv_coef()
    liner_omega = [0, 0]
    for sv, coef in itertools.zip_longest(wm_sv, wm_coef):
        liner_omega[0] += sv[1] * coef[0]
        liner_omega[1] += sv[2] * coef[0]
    liner_bias = liner_model.get_sv_rho()
    ax1, ax2 = plot_init()
    ax1.plot([-liner_bias/liner_omega[0], (liner_omega[1]+liner_bias)/(-liner_omega[0])], [0, 1])

    #2 高斯核SVM
    ## 1 求高斯核参数
    gaussian_model = svmutil.svm_train(labels, feature_spaces, '-h 0 -t 2 -c 1000 -q')
    wm_sv, wm_coef = gaussian_model.get_SV(), gaussian_model.get_sv_coef()
    gaussian_svs = [[feature_spaces[1], feature_spaces[2]] for feature_spaces in wm_sv]
    gaussian_bias = gaussian_model.get_sv_rho()
    gaussian_gamma = 0.5 #查看libsvm中，默认的γ为类别的倒数

    ## 2 求解决策函数边界（即结果为0的点）
    densitys = np.arange(0, 1, 0.01)
    decison_func = decision_gaussian(svs=gaussian_svs, coef=wm_coef, gamma=gaussian_gamma, bias=gaussian_bias)
    func = lambda x0: fsolve(lambda y0:decison_func( np.array([[x0, y0[0]], [x0, y0[1]]])), [0, 1])
    sugar_rates = list(map(func, densitys))

    pairs = map(lambda item:list(itertools.product([item[0]], item[1])), zip(densitys, sugar_rates))
    pairs = list(reduce(lambda density,sugar_rate:density+sugar_rate, pairs))
    '''
        pairs格式[(x1,x2),...]
    '''
    results = np.abs(decison_func(pairs))
    filtered_pairs = [pairs[i] for i in range(len(pairs)) if np.abs(results[i]) < 1e-3 ]
    '''
        由于scipy在求根时，无解的时候也会返回结果，故采取措施滤掉无效的结果
    '''
    ax2.plot(*list(zip(*filtered_pairs)), '.r')
    plt.show()

if __name__ == "__main__":
    main()
