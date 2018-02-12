#encoding:utf-8
'''
    AdaBoost算法，配合决策树桩
'''
import sys
from os import path
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import reduce
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )
from dataSet.watermelon_3alpha import wm_dataSet, watermelon_counterexample_x, watermelon_posiexam_x
from DecisionTreeStumps import generate_stumps, decide

POSI_COLOR = "#DAF9CA"
COUNTER_COLOR = "#C7B3E5"
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
    
    return ax1

def main():
    '''
        主函数
    '''
    #1 初始化权值和轮次
    Total_Turns =  8
    weights = [1/len(wm_dataSet)] * len(wm_dataSet)

    #2 Adaboosting
    def boosting(turns, packages, weights):
        '''
            尾递归实现boosting
            输入turns:剩余轮次，packages([权值alpha，分类器，分类边界等参数])，weights：当前权值
            输出packages
        '''
        if not turns:
            def final_classifier(x):
                '''
                    最终的组合分类器
                '''
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

    #3 求各个边界
    x_edges = [0, 1]
    y_edges = [0, 1]
    for var in l:
        if var[2][-1] == 0:
            x_edges.append(var[2][1])
        else:
            y_edges.append(var[2][1])
    x_edges = sorted(list(set(x_edges)))
    y_edges = sorted(list(set(y_edges)))
    '''
        去除重复边界并排序
    '''
    
    ax = plot_init()
    #4 求边界构成的最小矩形并且填色
    y_func = lambda y_pre_edge, y_post_edge:list(map(lambda x_pre_edge, x_post_edge:([x_pre_edge, y_pre_edge], [x_post_edge-x_pre_edge, y_post_edge-y_pre_edge])  , x_edges, x_edges[1:]))
    rects = list(reduce(lambda l1, l2:l1+l2, map(y_func, y_edges, y_edges[1:])))
    for rect in rects:
        color = POSI_COLOR if Adaboosting_classifier([rect[0][0] + rect[1][0]/2, rect[0][1] + rect[1][1]/2]) == 1 else COUNTER_COLOR
        ax.add_patch(
            patches.Rectangle(tuple(rect[0]),
             rect[1][0], 
             rect[1][1],
              color = color),
            
        )

    #5 绘制样本点并且绘制边界线（edges）
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax.plot(next(pose_points), next(pose_points), 'go')
    ax.plot(next(counter_points), next(counter_points), 'bo')#样本点
    for var in l:
        edge = var[2]
        if edge[-1] == 0:
            ax.plot([edge[1], edge[1]], [0, 1])
        else:
            ax.plot([0, 1], [edge[1], edge[1]])
    plt.show()
if __name__ == '__main__':  
    main()