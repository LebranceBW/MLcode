'''
西瓜数据集3.0阿尔法
x0 表示密度 x1表示含糖量 y = 1表示是好瓜，-1是烂瓜🍉
wm_dataset 格式(+1 为正例， -1为反例)
(
    ((x11,x12),y1),
    ((x21,x22),y2),
)

'''
from functools import reduce
watermelon_counterexample_x = (
    (0.666, 0.091),
    (0.243, 0.267),
    (0.245, 0.057),
    (0.343, 0.099),
    (0.639, 0.161),
    (0.657, 0.198),
    (0.360, 0.370),
    (0.593, 0.042),
    (0.719, 0.103),
)


watermelon_posiexam_x = (
    (0.697, 0.460),
    (0.774, 0.376),
    (0.634, 0.264),
    (0.608, 0.318),
    (0.556, 0.215),
    (0.403, 0.237),
    (0.481, 0.149),
    (0.437, 0.211),
)

watermelon_x = watermelon_posiexam_x+watermelon_counterexample_x

wm_dataSet = tuple(zip(watermelon_x, [1]*8 + [-1]*9))
wm_attridict = {"Density":0, "Sugar_rate":1}

def wm_picker(dataset, **kw):
    '''
        筛选器，这数据很鸡儿烦，弄个筛选器方便弄
        label,
    '''
    def decision_func(vector):
        def func(attri):
            if attri == 'label':
                return vector[1] == kw[attri]
            else:
                if attri in wm_attridict:
                    return vector[0][wm_attridict[attri]] in kw[attri]
                else:
                    raise RuntimeError("attribute is not in the dict！")
        return reduce(lambda x, y:x and y,map(func, kw))
    
    return tuple(zip(*filter(decision_func, zip(*dataset))))