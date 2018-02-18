#encoding:utf-8
'''
    KNN K近邻算法
'''
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_3alpha import wm_dataSet
from KDimensionTree import KDTree
import numpy as np
from MaxHeap import MaxHeap
import warnings
import matplotlib.pyplot as plt

def plot_init():
    '''
        初始化matplotlib
    '''
    ax1 = plt.subplots()[1]
    plt.title("KNN")
    plt.xlabel("Density")
    plt.ylabel("Sugar rate")
    return ax1

def generate_predict(trainning_set, neighbor_k=1, more_details=False):
    '''
        输入训练集，最近邻的点个数以返回判断函数
    '''
    #从数据集中训练出KD树
    kd_tree = KDTree.generate_tree(trainning_set)

    def distance_measure(point0, point1):
        '''
            # 距离度量函数，p=2 为欧氏距离， p=1为曼哈顿距离。
            # point: (x1, x2, x3, ...)
            # L = (Σ|point0_j - point1_j|**p) ** 1/p
            # 如果p是∞ L= max|point0_j - point1_j|
            返回point0， point1之间的欧式距离
        '''
        point0 = np.array(point0)
        point1 = np.array(point1)
        return np.sqrt(np.sum(np.square(point0 - point1)))

    def predict(feature):
        '''
            判断函数，输入特征值输出判断结果
        '''
        #1 随便从训练集中选择K个点作为K近邻,并组成最大值堆
        if not (neighbor_k % 2):
            warnings.warn("K值为偶数，可能会出现决策时正反样本数相等的情况")
        if neighbor_k > len(trainning_set) or neighbor_k < 0:
            raise RuntimeError("K值不合法")
        if neighbor_k > len(trainning_set)**0.5:
            warnings.warn("K值超过了样本总数的开方")
        dots = trainning_set[:neighbor_k]
        near_dots_heap = MaxHeap(map(lambda item: (distance_measure(item[0], feature), item), dots))

        #2 快速查找样本点所在最小区域，并将路过的样本点更新进入最大值堆
        def search(tree, near_dots_heap, stack=None):
            '''
                根据输入的KD树以及其划分策略，检索样本点所在区域（或者最近区域）。并将途经过的点入最大值堆并将其入栈
                tree：KD_tree
            '''
            if tree.is_leaf:
                '''
                    如果查找到了最接近样本点的叶节点，则开始回溯
                '''
                return near_dots_heap, stack
            if not stack:
                stack = []

            #如果该点到特征点的距离小于堆顶值，那么入堆
            distance = distance_measure(tree.dot[0], feature)
            if distance < near_dots_heap.max_key:
                near_dots_heap.pushpop((distance, tree.dot))

            #根据划分边界与策略，检索子树
            if feature[tree.axis] < tree.edge:
                stack.append(('left_tree', tree))
                return search(tree.left_tree, near_dots_heap, stack)
            stack.append(('right_tree', tree))
            return search(tree.right_tree, near_dots_heap, stack)

        near_dots_heap, stack = search(kd_tree, near_dots_heap)
        '''
            返回遍历栈以方便后面的回溯
        '''

        #3 回溯
        def review(stack, near_dots_heap):
            '''
                回溯，输入回溯栈，最大值堆
                返回最大值堆
            '''
            if not stack:
                return near_dots_heap

            radius = near_dots_heap.max_key #超球体的半径
            label, tree = stack.pop()
            if label == 'left_tree':
                subtree = tree.right_tree
            elif label == 'right_tree':
                subtree = tree.left_tree

            # 判断以feature为圆心，最大距离为半径的超球体是否与另一个子树有交割
            if (feature[tree.axis] - radius < tree.edge) and (feature[tree.axis] + radius > tree.edge):
                # 有的话检索另一子树中的最近点
                near_dots_heap = search(subtree, near_dots_heap)[0]
            return review(stack, near_dots_heap)

        #4 判断
        near_dots_heap = review(stack, near_dots_heap)
        result = 1 if sum(map(lambda item: item[1][1], near_dots_heap.to_list())) > 0 else -1

        if more_details:
        # 返回更多内容
            return result, near_dots_heap.to_list(), kd_tree
        return result

    return predict

def main():
    '''
        主函数
    '''
    predict_func = generate_predict(wm_dataSet, 3, True)
    feature = np.random.rand(2)
    predict_label, neighbors, kd_tree = predict_func(feature)
    ax1 = plot_init()
    kd_tree.draw_myself(ax1)

    for each in neighbors:
        ax1.plot([each[1][0][0], feature[0]], [each[1][0][1], feature[1]], color="m")
    ax1.plot(*feature, 'go' if predict_label > 0 else 'bo')
    plt.show()

if __name__ == '__main__':
    main()
