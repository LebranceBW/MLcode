#encoding:utf-8
#🍉西瓜书上的决策书例题
from dataSet.watermelon_2 import watermelon_attri
from dataSet.watermelon_2 import wm_trainningset
from dataSet.watermelon_2 import wm_validationset
from dataSet.watermelon_2 import watermelon_D
import numpy as np
from functools import reduce , partial
from mymodules.myclass import Tree

array_D = list(wm_trainningset)
array_Attri = [x for x in watermelon_attri][1 :-1]

def rate_category(D,value): #计算正反例的概率
    def func(D,value):#即Pk
        if D == []:
            return 0
        L = list(np.array(D).T[watermelon_attri[u'好坏']])
        return L.count(value)/L.__len__()
    if D == array_D:
        return [0,0.47058,0.52942,0][value]
    else:
        return func(D,value)  

def Dv(D,attri,value): #提取某一属性的数据集
    return list(filter(lambda unit:unit[watermelon_attri[attri]] == value,D))

def assembledGain(D,attri):
    #组合起来的信息增益
    def summulate(vector):#列表求和
        return reduce(lambda x,y: x+y,vector)

    def Ent(Pk):#信息熵
        def func(D):    
            def unit(i):
                temp = Pk(D,i)
                if temp != 0:
                    return temp * np.log2(temp)
                else:
                    return 0
            return -(unit(2)+unit(1)+unit(3))
        return func

    def sum_unit(Ent,Dv): #求和单元
        def func(D,attri):
            return lambda value:Dv(D,attri,value).__len__() * Ent(Dv(D,attri,value))
        return func

    def Gain(Ent,Dv): #信息增益
        def func(D,attri):
            return Ent(D) - summulate(map(sum_unit(Ent,Dv)(D,attri),[1,2,3]))/list(D).__len__()
        return func

    return Gain(Ent(rate_category),Dv)(D,attri)

def unit_Gain_test():
    D = list(watermelon_D)
    ans = [None,0.109,0.143,0.141,0.381,0.289,0.006]
    for x in watermelon_attri:
        if x == u"好坏" or x==u'编号':
            continue
        result = assembledGain(D,x)
        if(np.abs(result - ans[watermelon_attri[x]]) > 0.001):
            print(u"Failed:有关信息增益的单元测试失败,有关%s的测试结果为%f,正确结果为%f" % (x,result,ans[watermelon_attri[x]]))
            return False
        # else:
        #     print(u"%s 的信息增益为%.3f,正确结果为%.3f" % (x,result,ans[watermelon_attri[x]]))
    print("Passed:有关信息增益的单元测试成功")
    return True

def TreeGenerate(D,A):
    temp = rate_category(D,1)
    if  temp == 1 or temp == 0:
        return Tree(["坏瓜","好瓜"][int(temp)],True)
    elif A == []:
        return Tree(["坏瓜","好瓜"][int(temp + 0.5)],True)
    else:
        A = sorted(A,key=lambda x:assembledGain(D,x))
        node = Tree(A[-1])
        for i in [1,2,3]:
            if A[-1] == "触感" and i == 3:
                continue
            dv = Dv(D,A[-1],i)
            if dv == []:
                node.childTree[i-1] = Tree(["坏瓜","好瓜"][int(temp + 0.5)],True)
            else:
                node.childTree[i-1] = TreeGenerate(dv,A[:-1])
        return node

def accuracy(Tree,validate_set):
    def travel(subtree,unit):
        if subtree.isLeaf:
            return subtree.attri
        else:
            return travel(subtree[unit[watermelon_attri[subtree.attri]]],unit)
    
    compurefunc = lambda unit:[u'错误',u'好瓜',u'坏瓜'].index(travel(Tree,unit)) == unit[watermelon_attri[u'好坏']]
    return sum(map(compurefunc,validate_set)) / validate_set.__len__()

def main():
    unit_Gain_test()
    a = TreeGenerate(array_D,array_Attri)
    print("经过验证得到的准确率为%.2f" % accuracy(a,wm_validationset))
    print("决策树为" + a.__str__())
if __name__ == "__main__":
    main()