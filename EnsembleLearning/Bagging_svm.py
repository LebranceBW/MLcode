#encoding:utf-8
'''
    使用baggin算法提升线性svm以完成在西瓜数据集上的分类
'''
import numpy as np
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_3alpha import wm_dataSet, watermelon_counterexample_x, watermelon_posiexam_x
from lib.libsvm import svmutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import reduce
from itertools import product

TOTAL_TURNS = 8
POSI_COLOR = "#DAF9CA"
COUNTER_COLOR = "#C7B3E5"
def plot_init():
    '''
    初始化matplot，并将样本点绘制到ax中
    '''
    ax1 = plt.subplots()[1]
    ax1.set_title(u'Bagging')
    ax1.set_xlabel(u"Density")
    ax1.set_ylabel(u"Sugar Rate")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.7)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax1.plot(next(pose_points), next(pose_points), 'go')
    ax1.plot(next(counter_points), next(counter_points), 'bo')#样本点
    return ax1

def bootstrap_sampling(dataset):
    '''
        输入一个数据集，从数据集中随机选取元素组成与源数据集同规模的数据集（可以重复）
        那么约有63.2%的元素出现在新选出的数据集中
    '''
    rand_indexs = np.random.randint(0, len(dataset), len(dataset)) #生成一个随机索引下标
    return list(zip(*list(map(lambda index: dataset[index], rand_indexs))))


def main():
    '''
        主函数
    '''
    #1 bagging过程
    def bagging(turns, dataset, models=None):
        '''
            递归生成决策函数，
            输入：轮次， 数据集
            输出：bagging分类器， 详细的中间参数（（分类器，（错误率，分界点，分界策略,属性下标））
        '''
        if not models:
            models = list([])
        if not turns:
            def bagging_classifier(feature):
                '''
                    相对多数法。如果正反票数相同返回1
                    输入：特征向量
                    输出：+1 or -1
                '''
                result = np.sum(list(map(lambda model: svmutil.svm_predict([1]*len(feature), feature, model)[0], models)) ,axis=0)
                return list(map(lambda i: 1 if i>0 else -1, result))
            return bagging_classifier, models

        sampling_dataset = bootstrap_sampling(dataset)
        model = svmutil.svm_train(sampling_dataset[1], sampling_dataset[0], '-h 0 -t 0 -c 5 -q')
        models.append(model)
        return bagging(turns-1, dataset, models=models)

    def line_generate(model):
        '''
            根据model计算线与坐标轴的交点
        '''
        wm_sv, wm_coef = model.get_SV(), model.get_sv_coef()
        liner_omega = [0, 0]
        for sv, coef in zip(wm_sv, wm_coef):
            liner_omega[0] += sv[1] * coef[0]
            liner_omega[1] += sv[2] * coef[0]
        liner_bias = model.get_sv_rho()
        return [-liner_bias/liner_omega[0], (liner_omega[1]+liner_bias)/(-liner_omega[0])], [0, 1]

    bagging_classifier, svm_models = bagging(TOTAL_TURNS, wm_dataSet)
    print(bagging_classifier(watermelon_counterexample_x + watermelon_posiexam_x))
    ax = plot_init()
    lines = list(map(line_generate, svm_models))
    for line in lines:
        ax.plot(line[0], line[1])
    
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    dots = list(map(lambda i:list(i), product(x, y)))
    pack = list(zip(dots, bagging_classifier(dots)))
    for each in pack:
        ax.plot(each[0][0], each[0][1], '.r' if each[1]==1 else '.b')
    
 
    plt.show()
if __name__ == '__main__':
    main()
        