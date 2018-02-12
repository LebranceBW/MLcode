#encoding:utf-8
'''
    AdaBoost算法，配合决策树桩
'''
import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )
from dataSet.watermelon_3alpha import wm_dataSet, watermelon_counterexample_x, watermelon_posiexam_x
from DecisionTreeStumps import generate_stumps

def plot_init():
    '''
    初始化matplot，并将样本点绘制到ax中
    '''
    ax1 = plt.subplots()[1]
    ax1.set_title(u'AdaBoosting')
    ax1.set_xlabel(u"Density")
    ax1.set_ylabel(u"Sugar Rate")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.7)
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax1.plot(next(pose_points), next(pose_points), 'go')
    ax1.plot(next(counter_points), next(counter_points), 'bo')#样本点
    return ax1

def main():
    '''
        主函数
    '''
    Total_Turns = 11
    weights = [1/len(wm_dataSet)] * len(wm_dataSet)

    def boosting(turns, packages, weights):
        '''
            尾递归实现boosting
            输入turns:剩余轮次，packages([权值alpha，分类器，分类边界等参数])，weights：当前权值
            输出packages
        '''
        if not turns:
            def final_classifier(x):
                result = sum(map(lambda item:item[0]*item[1](x), packages))
                return 1 if result>0 else -1
            return packages, final_classifier
        classifier, pack = generate_stumps(wm_dataSet, weights) 
        error = pack[0]
        if error==0: return
        alpha = 0.5 * np.log((1-error)/error) #分类器权重
        weights = list(map(lambda sample, weigh:weigh*np.exp(-alpha*sample[1]*classifier(sample[0])), wm_dataSet, weights))
        factor = sum(weights)#归一化因子
        weights = list(map(lambda weigh:weigh/factor, weights))
        packages.append((alpha, classifier, pack))
        return  boosting(turns-1, packages, weights)

    l, Adaboosting_classifier = boosting(Total_Turns, [], weights)
    print(list(map(Adaboosting_classifier, watermelon_counterexample_x+watermelon_posiexam_x)))
    ax = plot_init()
    for var in l:
        edge = var[2]
        if edge[-1] == 0:
            ax.plot([edge[1], edge[1]], [0, 1])
        else:
            ax.plot([0, 1], [edge[1], edge[1]])
    plt.show()
if __name__ == '__main__':  
    main()