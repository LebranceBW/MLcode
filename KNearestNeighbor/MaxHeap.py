#encoding:utf-8
'''
    用python自带的最小值堆构建无重复内容的最大值堆
    简单的把键值翻转并检查是否重复
'''
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
        if item in self.__heap:
            return
        return heapq.heappush(self.__heap, item)

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
        if item in self.__heap:
            return
        return heapq.heappushpop(self.__heap, item)

    @property
    def max_key(self):
        '''
            返回堆顶的元素键值
        '''
        return -self.__heap[0][0]

    def to_list(self):
        '''
            将堆中的数据转换为列表
        '''
        return list(map(lambda item: (-item[0], item[1]), self.__heap))
        