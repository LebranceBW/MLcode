#encoding:utf-8
'''
    接受带权值的数据集并实现一个决策树桩🌲
'''
from functools import partial
from enum import Enum
class Decide(Enum):
    '''
    笔记：decide.greater_to_positive 也是一个对象！不能直接用来和值进行比较
        Decide.greater_to_positive.value 返回其实际值
    '''
    greater_to_positive = True
    lesser_to_positive = False

def generate_stumps(dataset, weights=None):
    '''
        生成决策树桩，输入dataset格式为：
        [
            [[x11,x12],y1],
            [[x21,x22],y2]
            ...
        ]
        weights 为输入的权值
        [w1, w2, w3...]
        返回：
        （分类器，（错误率，分界点，分界策略,属性下标））
    '''
    if weights == None:
        weights = [1/len(dataset)] * len(dataset)

    def each_attri_best(i, dataset):
        '''
            返回某一属性中最佳的错误率和分类策略，i为属性下标，dataset为zipped_set
            返回格式：(错误率，分界点，分界策略,属性下标)
        '''
        itemset = ([item[0][i], item[1], item[2]] for item in dataset)
        '''
            格式[xi ,y, weight]
        '''
        sorted_dataset = sorted(itemset, key=lambda item: item[0])
        divide_points = map(lambda x, y: float(format((x[0]+y[0])/2, ".3f")), sorted_dataset, sorted_dataset[1:])#求相邻两属性值的中间值,移位
        error_rates = map(partial(each_dividepoint, dataset=sorted_dataset), divide_points)
        local_best = sorted(error_rates, key=lambda item: item[0])[0]
        return (*local_best, i)

    def each_dividepoint(point, dataset):
        '''
            dataset 为数据集（sorted_dataset)，point为一个点
            返回各个分类点的带权误差
            默认认为大于临界为正例（greater_to_positive)
            返回值为（错误率，分界点，分界策略）
        '''
        # def xor(a, b):
        #     #异或逻辑
        #     if bool(a) != bool(b): return True
        #     return False
        error_rate = sum(map(lambda item: item[2]*(1 if (item[0] < point) != (item[1] == -1) else 0), dataset))
        '''
            如果该元素在临界点左侧且为正例则统计其权值，在临界点右侧且为反例的同理
        '''
        if error_rate > 0.5:#如果错误率大于0.5，那么说明分类策略反了
            return (1-error_rate, point, Decide.lesser_to_positive)
        return (error_rate, point, Decide.greater_to_positive)
    zipped_dataset = list(map(lambda data, weight: (*data, weight), dataset, weights))
    # [[*data, weight] for data in dataset for weight in weights]
    '''
      把数据集和权值打包 zipped_dataset 格式  [[x11,x12],y1,w1],...

    '''
    attri_numbers = len(dataset[0][0])
    best_classifier_pack = sorted(map(partial(each_attri_best, dataset=zipped_dataset), range(attri_numbers)), key=lambda item: item[0])[0]
    '''
        pack格式：(错误率，分界点，分界策略,属性下标)
    '''

    def weak_classifier(feature_vector):
        '''
            用于返回的弱分类函数
            输入特征向量，输出标签
        '''
        if (feature_vector[best_classifier_pack[-1]] > best_classifier_pack[1]) == (best_classifier_pack[2].value):
            return 1
        return -1

    return weak_classifier, best_classifier_pack

def test_stumps(dataset):
    '''
        单元测试模块，TODO
    '''
    print(generate_stumps(dataset, [1/17]*17))

if __name__ == '__main__':
    print("这只是个模块")
