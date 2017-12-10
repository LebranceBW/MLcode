#encoding:utf-8
#🍉西瓜书上的决策书例题
from dataSet.watermelon_2 import watermelon_D
from dataSet.watermelon_2 import watermelon_attri
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce,partial

array_D = list(watermelon_D)
array_Attri = [x for x in watermelon_attri][:-1]

def rate_category(D,attri,value): #计算正反例的概率
    def func(D,attri,value):#即Pk
        if D == []:
            return 0
        L = list(np.array(D).T[watermelon_attri[attri]])
        return L.count(value)/L.__len__()
    if (D,attri) == (array_D,u"好坏"):
        return [0,0.47058,0.52942,0][value]
    else:
        return func(D,attri,value)  

def Dv(D,attri,value): #提取某一属性的数据集
    return list(filter(lambda unit:unit[watermelon_attri[attri]] == value,D))

def assembledGain(D,attri):
    #组合起来的信息增益
      

    def summulate(vector):#列表求和
        return reduce(lambda x,y: x+y,vector)

    def Ent(Pk):#信息熵
        def func(D):    
            def unit(i):
                temp = Pk(D,u"好坏",i)
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
    ans = [0.109,0.143,0.141,0.381,0.289,0.006]
    for x in watermelon_attri:
        if x == u"好坏":
            continue
        result = assembledGain(D,x)
        if(np.abs(result - ans[watermelon_attri[x]]) > 0.001):
            print(u"有关Gain()的单元测试失败,有关%s的测试结果为%f,正确结果为%f" % (x,result,ans[watermelon_attri[x]]))
            return False
        else:
            print(u"%s 的信息增益为%.3f,正确结果为%.3f" % (x,result,ans[watermelon_attri[x]]))
    print("有关Gain()的单元测试成功")
    return True

class Tree:
    def __init__(self):
        self.__attri = ""
        self.__list = [None,None,None]
        self.__isLeaf = False
    def __init__(self,attri,isLeaf = False):
        self.__attri = attri
        self.__list = [None,None,None]
        self.__isLeaf = isLeaf
    @property
    def childTree(self):
        return self.__list
    @childTree.setter
    def childTree(self,value):
        self.__list = value

    @property
    def isLeaf(self):
        return self.__isLeaf

    @property
    def attri(self):
        return self.__attri

    def __str__(self):
        def travel(node,depth = 0,L=[]):
            if(node == None):
                return
            elif(node.isLeaf):
                try:
                    L[depth].append(node.attri)
                except IndexError:
                    L.append([node.attri])
                return
            else:
                try:
                    L[depth].append(node.attri)
                except IndexError:
                    L.append([node.attri])
                travel(node.childTree[0],depth+1,L)
                travel(node.childTree[1],depth+1,L)
                travel(node.childTree[2],depth+1,L)
        L = []
        string = "决策树为：{\n "
        travel(self,0,L)
        for x in L:
            for y in x:
                string = string + y + " "
            string += "\n "
        return string + "}"


def TreeGenerate(D,A):
    temp = rate_category(D,u"好坏",1)
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
def main():
    unit_Gain_test()
    a = TreeGenerate(array_D,array_Attri)
    print(a)
if __name__ == "__main__":
    main()