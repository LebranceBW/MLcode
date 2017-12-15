#encoding:utf-8
#🍉使用基尼指数判断的决策树
from dataSet.watermelon_2 import wm_trainningset as train_set
from dataSet.watermelon_2 import wm_validationset as validate_set
from dataSet.watermelon_2 import watermelon_attri
from dataSet.watermelon_2 import watermelon_D
from mymodules.myclass import Tree
import numpy as np
from functools import reduce,partial
array_Attri =[x for x in watermelon_attri][1 :-1]
def rate_category(D,value): #计算正反例的概率
    def func(D,value):#即Pk
        if not D:
            return 0
        L = list(np.array(D).T[watermelon_attri[u'好坏']])
        return L.count(value)/L.__len__()
    if D == watermelon_D:
        return [0,0.47058,0.52942,0][value]
    else:
        return func(D,value)  

def test_Giniattri(D,array_Attri):
    ans = [None,0.35,0.44,0.40,0.40,0.35,0.50]
    for attri in array_Attri:
        temp = Gini_attri(D,attri)
        if temp == ans[watermelon_attri[attri]]:
            pass
        else:
            print("Failed: 基尼指数单元测试失败，%s的基尼指数%f计算错误，结果应该为%f" % (attri,temp,ans[watermelon_attri[attri]]))
            return False
        print("Passed: 基尼指数单元测试通过")
        return True
       

def Dv(D,attri,value): #提取某一属性的数据集
    return list(filter(lambda unit:unit[watermelon_attri[attri]] == value,D))



def Gini_attri(D,attri): #属性a的Gini系数
    def Gini(pk): #基尼指数，反应了随机从样本中抽取两个样本其标记不同的概率
        return lambda D:1 - pk(D,2)**2 - pk(D,1)**2

    def Gini_index(Gini,Dv): 
        def func(D,attri):
            return  sum(map(lambda value:Dv(D,attri,value).__len__()*Gini(rate_category)(Dv(D,attri,value)),[1,2,3]))/ D.__len__()
        return func
    return Gini_index(Gini,Dv)(D,attri)
    
def TreeGenerate(D,A,weigh_fun):
    temp = rate_category(D,1)
    if  temp == 1 or temp == 0:
        return Tree(["坏瓜","好瓜"][int(temp)],True)
    elif A == []:
        return Tree(["坏瓜","好瓜"][int(temp + 0.5)],True)
    else:
        A = sorted(A,key=lambda x:weigh_fun(D,x))
        node = Tree(A[-1])
        for i in [1,2,3]:
            if A[-1] == "触感" and i == 3:
                continue
            dv = Dv(D,A[-1],i)
            if dv == []:
                node.childTree[i-1] = Tree(["坏瓜","好瓜"][int(temp + 0.5)],True)
            else:
                node.childTree[i-1] = TreeGenerate(dv,A[:-1],weigh_fun)
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
    test_Giniattri(train_set,array_Attri)
    a = TreeGenerate(train_set,array_Attri,Gini_attri)
    print(a)
    print(accuracy(a,validate_set))
if __name__ == "__main__":
    main()