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
import heapq

class MaxHeap:
    '''
        封装最大值堆,方法是将权值取相反数
        权值必须为数
    '''
    def __init__(self, iterable):
        '''
            使用列表初始化heap
            [(<key>, content),...]
            [(5, 'write code'), ...]
        '''
        array = list(map(lambda item: (-item[0], item[1]), iterable))
        heapq.heapify(array)
        self.__heap = array

    def __len__(self):
        '''
            重载长度方法
        '''
        return len(self.__heap)

    def push(self, item):
        '''
            item格式
            (<key>, content)
        '''
        item = (-item[0], item[1])
        heapq.heappush(self.__heap, item)

    def pop(self):
        '''
            堆弹出
        '''
        if not self.__heap:
            raise RuntimeError("Empty Heap")
        item = heapq.heappop(self.__heap)
        return (-item[0], item[1])

    def pushpop(self, item):
        '''
           先入堆，再出堆
        '''
        item = (-item[0], item[1])
        return heapq.heappushpop(self.__heap, item)

    def top_item(self):
        '''
            返回堆顶数据但是不弹出
        '''
        return (-heapq.nlargest(1, self.__heap)[0][0], heapq.nlargest(1, self.__heap)[0][1])

    def to_list(self):
        return self.__heap

def generate_predict(trainning_set, neighbor_k=1):
    '''
        输入训练集，最近邻树以返回判断函数
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
        if neighbor_k > len(trainning_set):
            raise RuntimeError("K值超过了样本点的数量！")
        elif neighbor_k > len(trainning_set)**0.5:
            print("推荐K值不超过训练集数目的开方")
        dots = trainning_set[:neighbor_k]
        max_heap = MaxHeap(map(lambda item: (distance_measure(item[0], feature), item), dots))

        #2 快速查找样本点所在最小区域，并将路过的样本点更新进入最大值堆
        def point_in_area(point, area):
            '''
                判断点是否在区域内
                point：(x1 ,y1, z1, ...)
                area ((xmin, ymin, zmin...), (xmax, ymax, zmax))
            '''
            for i in enumerate(point):
                if (i[1] < area[0][i[0]]) or (i[1] > area[1][i[0]]):
                    return False
            return True

        def search(tree, max_heap, stack=None):
            '''
                输入根节点与初始化了的最大值堆，返回一个部分优化了的最大值堆与遍历栈
            '''
            if tree.is_leaf:
                '''
                    如果查找到了最接近样本点的叶节点，则开始回溯
                '''
                return max_heap, stack
            if not stack:
                stack = []
            distance = distance_measure(tree.dot[0], feature)
            if distance < max_heap.top_item()[0]: #如果该点到特征点的距离小于堆顶值，那么入堆
                max_heap.pushpop((distance, tree.dot))
            if point_in_area(feature, tree.left_tree.area): #如果该点在左子树区域内
                stack.append((tree, 'left'))
                return search(tree.left_tree, max_heap)
            stack.append((tree, 'right'))
            return search(tree.right_tree, max_heap)

        max_heap, stack = search(kd_tree, max_heap)
        '''
            返回遍历栈以方便后面的回溯
        '''

        #3 回溯
        

        def cross_point_area(point, area, radius):
            '''
                是否发生交割
                有很大BUG！
            '''
            for i in enumerate(point):
                if (i[1] < area[0][i[0]]) and (i[1]+radius > area[0][i[0]]):
                    return True
                if (i[1] > area[1][i[0]]) and (i[1]-radius < area[1][i[0]]):
                    return True
            return False
        
        def travel(tree, points=None):
            '''
                遍历一个树并返回其所有的样本点
            '''
            if not points:
                points = []
            points.append(tree.dot)   
            if tree.is_leaf:
                return points
            points = travel(tree.left_tree, points)
            points = travel(tree.right_tree, points)
            return points

        def review(stack, max_heap):
            if not stack:
                return max_heap
            radius = max_heap.top_item[0] #最大距离    
            tree, label = stack.pop()
            if label == 'left':
                subtree = tree.right_tree
            else:
                subtree = tree.left_tree
            if cross_point_area(feature, subtree.area, radius): #如果发生了交割
                points = travel(subtree)
                for each in points:
                    distance = distance_measure(each[0], feature)
                    if distance < max_heap.top_item[0]: #如果该点到特征点的距离小于堆顶值，那么入堆
                        max_heap.pushpop((distance, each))
            review(stack ,max_heap)

        #4 判断
        return 1 if sum(map(lambda item: item[1][1], max_heap.to_list()))>0 else -1
    
    return predict

def main():
    predict_func = generate_predict(wm_dataSet, 3)
    print(list(map(lambda item: predict_func(item[0]), wm_dataSet)))
if __name__ == '__main__':
    main()