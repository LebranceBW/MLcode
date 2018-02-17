#encoding:utf-8
'''
    数据格式：wm开头的格式为[[<feature>],[<label>]]
'''
from functools import reduce

raw = (
    ((1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 1.0),
    ((2, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0), 1.0),
    ((3, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0), 1.0),
    ((4, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0), 1.0),
    ((5, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0), 1.0),
    ((6, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0), 1.0),
    ((7, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0), 1.0),
    ((8, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0), 1.0),
    ((9, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0), 0.0),
    ((10, 1.0, 3.0, 3.0, 1.0, 3.0, 2.0), 0.0),
    ((11, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0), 0.0),
    ((12, 3.0, 1.0, 1.0, 3.0, 3.0, 2.0), 0.0),
    ((13, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0), 0.0),
    ((14, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0), 0.0),
    ((15, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0), 0.0),
    ((16, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0), 0.0),
    ((17, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0), 0.0))

def Subtract(A, B):  
    '''
        A - B
    '''
    return [x for x in A if x not in B]
serials = (1,2,3,6,7,10,14,15,16,17)

def wm_counter(dataset):
    if not dataset:
        return 0
    return len(dataset[0])

def wm_picker(dataset, **kw):
    '''
        筛选器，这数据很鸡儿烦，弄个筛选器方便弄
        输入格式：数据集， “属性名”=属性值
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
    
    #下面代码是错误的❌,filter有个特性是惰性求值！！！！！！！
    # for attri in kw:
    #     if attri is 'label':
    #         templist = list(filter(lambda vector:vector[1] == kw[attri], templist))
    #     else:
    #         templist = list(filter(lambda vector:vector[0][watermelon_attri[attri]] in kw[attri], templist))
    # return list(func(dataset, iter(kw)))
wm_attridict = {u"编号":0, u"色泽":1, u"根蒂":2, u"敲声":3, u"纹理":4, u"脐部":5, u"触感":6}

wm_dataset =  tuple(zip(*raw))
wm_trainningset = wm_picker(wm_dataset, 编号 = serials)
wm_validationset = wm_picker(wm_dataset, 编号 = Subtract(range(18), serials))

