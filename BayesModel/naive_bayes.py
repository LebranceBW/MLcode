#encoding:utf-8
'''
    经过laplace修正的朴素贝叶斯分类器
    应用在西瓜数据集2.0上 🍒
'''
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_2 import wm_trainningset, wm_validationset, wm_picker, wm_counter, wm_attridict
import numpy as np


def decision_table_generate(dataset):
    '''
        生成频率表
    '''
    def laplace_probability(dataset, totalcount, **kw):
        '''
            经过拉普拉斯修正的概率函数
        '''
        return (wm_counter(wm_picker(dataset, **kw))+1) / (wm_counter(dataset)+totalcount)

    probaility_table = [[], []]
    positive_dataset = wm_picker(dataset, label=1)
    negative_dataset = wm_picker(dataset, label=0)
    for var in wm_attridict:
        if var == u'编号':
            probaility_table[0].append([])
            probaility_table[1].append([])
            continue
        if var == u'触感': #触感就两个取值
            probaility_table[1].append([laplace_probability(positive_dataset, 2, **{var:[x]}) for x in range(1, 3)])
            probaility_table[0].append([laplace_probability(negative_dataset, 2, **{var:[x]}) for x in range(1, 3)])
        else:
            probaility_table[1].append([laplace_probability(positive_dataset, 3, **{var:[x]}) for x in range(1, 4)])
            probaility_table[0].append([laplace_probability(negative_dataset, 3, **{var:[x]}) for x in range(1, 4)])
    return tuple(probaility_table)

def predict_vectors(decision_table, vectors):
    '''
        对样本群来预测结果
    '''
    log_probaility_func = lambda attri_var, attri_p: np.log(attri_p[int(attri_var-1)])
    def predict_vector(vector):
        '''
            对每个样本预测结果
        '''
        posi_rate = sum(map(log_probaility_func, vector[1:], decision_table[1][1:]))
        nege_rate = sum(map(log_probaility_func, vector[1:], decision_table[0][1:]))
        return int(posi_rate > nege_rate)
    return tuple(map(predict_vector, vectors))

def main():
    '''
        主函数
    '''
    table = decision_table_generate(wm_trainningset)
    print(wm_trainningset[1], predict_vectors(table, wm_trainningset[0]))
    print(wm_validationset[1], predict_vectors(table, wm_validationset[0]))


if __name__ == '__main__':
    main()
