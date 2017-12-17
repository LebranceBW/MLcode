#encoding:utf-8
#🍉使用基尼指数判断的决策树
#贪心是指尽可能使正确率更高，若是划分前后正确率相同则尽量划分
#非贪心则是指正确率没有提升则不划分
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
    
def rawTreeGenerate(D,A,weigh_fun):
    temp = rate_category(D,1)
    if  temp == 1 or temp == 0 or A == []:
        return Tree(["坏瓜","好瓜"][int(temp)],True)
    else:
        A = sorted(A,key=lambda x:weigh_fun(D,x))
        node = Tree(A[-1])
        def iterator_func(node,i = 1):#替代掉循环
            if i == 4 or (A[-1] == "触感" and i == 3): 
                return
            dv = Dv(D,A[-1],i)
            if dv == []:
                node[i] = Tree(["坏瓜","好瓜"][int(temp + 0.5)],True)
            else:
                node[i] = rawTreeGenerate(dv,A[:-1],weigh_fun)
            iterator_func(node,i+1)

        iterator_func(node)
        return node

def prePruneTree(D,A,weigh_fun,isgreedy=False,node=None,root=None,accuracy=0):
    majority = lambda D:Tree([u"坏瓜",u"好瓜"][int(rate_category(D,1))],True) #返回集合中大多数元素所属类型的节点
    def unfold(node,attri,i=1):
        if i==4:
            node.attri = attri
            node.isLeaf = False
            return
        dv = Dv(D,attri,i)
        if attri == u"触感" and i == 3:
            node[i] = None
        elif dv == []:
            node[i] = majority(D)
        else:
            node[i] = majority(dv)
        unfold(node,attri,i+1)
    
    if root==None and node==None:
        node=root=Tree(u"好瓜") #初始化

    temprate = rate_category(D,1)
    if A == [] or temprate == 1 or temprate == 0:
        return root

    A = sorted(A,key=lambda x:weigh_fun(D,x))
    temp = node #为node做一下备份
    unfold(node,A[-1])
    cur_accuracy = accuracy_fun(root,validate_set)

    if isgreedy:
        if cur_accuracy < accuracy:     #尽量划分使得准确率最高，但是正确做法是减少划分次数
            node = temp
            return root
        else:
            prePruneTree(Dv(D,A[-1],1),A[:-1],weigh_fun,isgreedy,node[1],root,cur_accuracy)
            prePruneTree(Dv(D,A[-1],2),A[:-1],weigh_fun,isgreedy,node[2],root,cur_accuracy)
            prePruneTree(Dv(D,A[-1],3),A[:-1],weigh_fun,isgreedy,node[3],root,cur_accuracy)
    else:
        if cur_accuracy <= accuracy:     #尽量划分使得枝桠最少
            node = temp
            return root
        else:
            prePruneTree(Dv(D,A[-1],1),A[:-1],weigh_fun,isgreedy,node[1],root,cur_accuracy)
            prePruneTree(Dv(D,A[-1],2),A[:-1],weigh_fun,isgreedy,node[2],root,cur_accuracy)
            prePruneTree(Dv(D,A[-1],3),A[:-1],weigh_fun,isgreedy,node[3],root,cur_accuracy)
    return root
     


def accuracy_fun(Tree,validate_set):
    def travel(subtree,unit):
        if subtree.isLeaf:
            return subtree.attri
        else:
            return travel(subtree[unit[watermelon_attri[subtree.attri]]],unit)
    
    compurefunc = lambda unit:[u'错误',u'好瓜',u'坏瓜'].index(travel(Tree,unit)) == unit[watermelon_attri[u'好坏']]
    return sum(map(compurefunc,validate_set)) / validate_set.__len__()
def main():
    test_Giniattri(train_set,array_Attri)
    a = rawTreeGenerate(train_set,array_Attri,Gini_attri)
    b = prePruneTree(train_set,array_Attri,Gini_attri)
    c = prePruneTree(train_set,array_Attri,Gini_attri,True)
    print("未剪枝的决策树"+a.__str__())
    print("非贪心预剪枝的决策树"+b.__str__())
    print("贪心预剪枝的决策树"+c.__str__())
    print("未剪枝的决策树正确率为：%.3f" % accuracy_fun(a,validate_set))
    print("非贪心预剪枝的决策树正确率为：%.3f" % accuracy_fun(b,validate_set))
    print("贪心预剪枝的决策树正确率为：%.3f" % accuracy_fun(c,validate_set))
if __name__ == "__main__":
    main()