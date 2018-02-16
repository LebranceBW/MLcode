#encoding:utf-8
'''
    使用baggin算法提升决策树桩以完成在西瓜数据集上的分类
'''
import numpy as np
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_3alpha import wm_dataSet, watermelon_counterexample_x, watermelon_posiexam_x
from DecisionTreeStumps_Model import generate_stumps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import reduce

TOTAL_TURNS = 20
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
    return ax1

def bootstrap_sampling(dataset):
    '''
        输入一个数据集，从数据集中随机选取元素组成与源数据集同规模的数据集（可以重复）
        那么约有63.2%的元素出现在新选出的数据集中
    '''
    rand_indexs = np.random.randint(0, len(dataset), len(dataset)) #生成一个随机索引下标
    return list(map(lambda index: dataset[index], rand_indexs))


def main():
    '''
        主函数
    '''

    #1 bagging过程
    def bagging(turns, dataset, package=None):
        '''
            递归生成决策函数，
            输入：轮次， 数据集
            输出：bagging分类器， 详细的中间参数（（分类器，（错误率，分界点，分界策略,属性下标））
        '''
        if not package:
            package = list([])
        if not turns:
            def bagging_classifier(feature):
                '''
                    相对多数法。如果正反票数相同返回1
                    输入：特征向量
                    输出：+1 or -1
                '''
                result = sum(map(lambda pack: pack[0](feature), package))
                return 1 if result > 0 else -1
            return bagging_classifier, package

        pack = generate_stumps(bootstrap_sampling(dataset))
        package.append(pack)
        return bagging(turns-1, dataset, package=package)

    bagging_classifier, detail_info_package = bagging(TOTAL_TURNS, wm_dataSet)
    print(list(map(lambda sample: bagging_classifier(sample[0]), wm_dataSet)))

    #2 求边界
    ax = plot_init()
    x_edges = [0, 1]
    y_edges = [0, 1]
    for var in detail_info_package:
        if var[1][-1] == 0:
            x_edges.append(var[1][1])
            ax.plot([x_edges, x_edges], [0, 1])
        else:
            y_edges.append(var[1][1])
            ax.plot([0, 1], [y_edges, y_edges])
            
    x_edges = sorted(list(set(x_edges)))
    y_edges = sorted(list(set(y_edges)))
    '''
        去重并排序
    '''

    #3 求边界构成的最小矩形并且填色
    def y_func(y_pre_edge, y_post_edge): 
        return list(map(lambda x_pre_edge, x_post_edge: ([x_pre_edge, y_pre_edge], [x_post_edge-x_pre_edge, y_post_edge-y_pre_edge]), x_edges, x_edges[1: ]))

    rects = list(reduce(lambda l1, l2: l1+l2, map(y_func, y_edges, y_edges[1: ])))
    for rect in rects:
        color = POSI_COLOR if bagging_classifier([rect[0][0] + rect[1][0]/2, rect[0][1] + rect[1][1]/2]) == 1 else COUNTER_COLOR
        ax.add_patch(
            patches.Rectangle(tuple(rect[0]), rect[1][0], rect[1][1], color=color),
        )

    #4 绘制样本点并且绘制边界线（edges）
    pose_points, counter_points = zip(*watermelon_posiexam_x), zip(*watermelon_counterexample_x)
    ax.plot(next(pose_points), next(pose_points), 'go')
    ax.plot(next(counter_points), next(counter_points), 'bo')#样本点

    # # print(list(map(lambda info:(info[1],info[2]), detail_info_package)))
    # for i in detail_info_package:
    #     print(i[1], i[2])
    plt.show()
if __name__ == '__main__':
    main()
        