#encoding:utf-8
'''
    太难写了，先鸽
    专用来处理连续值的决策树，用信息增益作为判定条件
    西瓜数据集3.0α
    dataset 格式：
    [[x11, x12], y1],[[x21, x22],y2]...
'''
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_3alpha import wm_dataSet, wm_attridict, wm_picker
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class Decide(Enum):
    greater_to_positive = True
    lesser_to_positive = False

def infomation_gain(dataset, attri_name):
    '''
        输入：数据集， 属性值（Density， Sugar_rate）
        输出：信息增益，分界点，属性名称
    '''

    def rate_category_func(dataset, label_val):
        '''
            求数据集中某一标签样本的比例
            输入label格式为np.array([label ...])
        '''
        if not dataset:
            return 0
        return list(map(lambda label: len(wm_picker(dataset, label=label))/len(dataset)), label_val)

    def info_entropy_func(rate_category):
        """
            entropy['ɛntrəpi]:熵
            输入分类占比函数， 求其信息熵
        """
        def func(dataset):
            '''
                输入数据集
            '''
            label_vals = np.array([-1, 1])
            return np.sum(-rate_category(dataset, label_vals) * np.log2(rate_category(dataset, label_vals)))
        return func

    def each_attri_val_gain(ent):    
        '''
            输入信息熵
        '''
        def func(dataset, attri_name):
            '''
                输入数据集，属性名称，属性值
                输出 当前属性的最大信息增益与之对应的分界点
            '''

            def sub_expression(dataset, index):
                '''
                    输入子数据集
                    输出 Σ|Dt| * Ent(Dt) / |D|
                '''
                return sum(map(lambda subdataset: len(subdataset)*ent(subdataset)/len(dataset), [dataset[:index], dataset[index:]]))
            
            dataset = sorted(dataset, key=lambda sample:sample[0][wm_attridict[attri_name]])
            gains = map(lambda i: [ent(dataset) - sub_expression(dataset, i), i], range(len(dataset)))
            max_gain, max_index = sorted(gains, key=lambda item:item[0])
            divide_point = (dataset[max_index - 1][0][wm_attridict[attri_name]] + dataset[max_index][0][wm_attridict[attri_name]]) / 2
            return max_gain, divide_point, attri_name
        return func

    return each_attri_val_gain(info_entropy_func(rate_category_func))(dataset, attri_name) 



def main():
if __name__ == "__main__":
    main()