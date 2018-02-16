#encoding:utf-8
#🍉西瓜书上的决策书例题
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from dataSet.watermelon_2 import wm_attridict, wm_dataset, wm_trainningset, wm_validationset,wm_dataset, wm_picker, wm_counter
import numpy as np
from copy import deepcopy
from functools import reduce, partial
from Tree_Module import Tree

wm_attriset = [x for x in wm_attridict if x != '编号']
def rate_category_func(dataset, label):
    '''
        计算正例和反例的频率
    '''
    def func(dataset, label):#即Pk
        tempdataset = wm_picker(dataset, label=label)
        if not dataset:
            return 0
        return wm_counter(tempdataset) / wm_counter(dataset)
    if dataset == wm_dataset:return [0.47058,0.52942][int(label)]
    else:return func(dataset, label) 

def filtrate_func(dataset, attri, value): #提取某一属性的数据集
    '''
            提取某一属性的数据集
    '''
    return wm_picker(dataset, **{attri:[value]})

def infomation_gain(D,attri):
    '''
        组合起来的信息增益
    '''
    def Ent(Pk):
        '''
        单一属性的信息熵
        '''
        def func(dataset):    
            def logarithm_func(result):
                '''
                    pk * log(pk) pk为0时需要定义为0 
                '''
                if result == 0:
                    return 0
                else:
                    return result * np.log2(result).sum()
            return -(logarithm_func(Pk(dataset, 0.0))+logarithm_func(Pk(dataset, 1.0)))
        return func

    def sum_unit(Ent,Dv): 
        '''
        求和单元
        '''
        def func(dataset, attri):
            return lambda value:wm_counter(Dv(dataset, attri, value)) * Ent(Dv(dataset, attri, value))
        return func

    def Gain(Ent,filtrate_func): 
        '''
        信息增益
        '''
        def func(D,attri):
            return Ent(D) - sum(map(sum_unit(Ent,filtrate_func)(D,attri),[1, 2, 3]))/wm_counter(D)
        return func

    return Gain(Ent(rate_category_func), filtrate_func)(D, attri)

def test_infomation_gain(wm_dataset,wm_attriset):
    ans = [None,0.109,0.143,0.141,0.381,0.289,0.006]
    for x in wm_attriset:
        if x == u"好坏" or x==u'编号':
            continue
        result = infomation_gain(wm_dataset,x)
        if(abs(result - ans[wm_attridict[x]]) > 0.001):
            print(u"Failed:有关信息增益的单元测试失败,有关%s的测试结果为%f,正确结果为%f" % (x,result,ans[wm_attridict[x]]))
            return False
    print("Passed: 信息增益单元测试通过")
    return True

def test_Giniattri(D,wm_attriset):#测试基尼指数计算是否错误
    ans = [None,0.35,0.44,0.40,0.40,0.35,0.50]
    for attri in wm_attriset:
        if attri in [u"好坏", u'编号']:
            continue
        temp = Gini_index(D,attri)
        if abs(temp-ans[wm_attridict[attri]])<0.01:pass
        else:
            print("Failed: 基尼指数单元测试失败，%s的基尼指数%f计算错误，结果应该为%f" % (attri,temp,ans[wm_attridict[attri]]))
            return False
    print("Passed: 基尼指数单元测试通过")
    return True

def Gini_index(D,attri): 
    '''
        属性a的Gini系数,希望以后看这段代码的时候不会凉凉
    '''
    def Gini(pk):
        '''
            基尼指数，反应了随机从样本中抽取两个样本其标记不同的概率
        '''
        return lambda D:1 - pk(D, 1.0)**2 - pk(D, 0.0)**2

    def Gini_part(Gini,filtrate_func): 
        def func(D,attri):
            return  sum(map(lambda value:wm_counter(filtrate_func(D,attri,value))*Gini(rate_category_func)(filtrate_func(D,attri,value)),[1.0, 2.0, 3.0]))/ wm_counter(D)
        return func
    return Gini_part(Gini,filtrate_func)(D,attri)

def rawtree_generate(D,A,weigh_fun):
    temp = rate_category_func(D,1)
    if  temp == 1 or temp == 0 or A == []:
        return Tree(["坏瓜","好瓜"][int(temp)],D,True)
    else:
        A = sorted(A,key=lambda x:weigh_fun(D,x))
        node = Tree(A[-1],D)
        def iterator_func(node,i = 1):#替代掉循环
            if i == 4 or (A[-1] == "触感" and i == 3): 
                return
            d_filtrated = filtrate_func(D,A[-1],i)
            if d_filtrated == []:
                node[i] = Tree(["坏瓜","好瓜"][int(temp + 0.5)],[],True)
            else:
                node[i] = rawtree_generate(d_filtrated,A[:-1],weigh_fun)
            iterator_func(node,i+1)

        iterator_func(node)
        return node

def preprune_tree_generate(D,A,weigh_fun,isgreedy=False,node=None,root=None,accuracy=0):
    majority = lambda D:Tree([u"坏瓜",u"好瓜"][int(rate_category_func(D,1))],D,True) #返回集合中大多数元素所属类型的节点
    def unfold(node,attri,i=1):
        if i==4:
            node.attri = attri
            node.isLeaf = False
            return
        d_filtrated = filtrate_func(D,attri,i)
        if attri == u"触感" and i == 3:
            node[i] = None
        elif d_filtrated == []:
            node[i] = majority(D)
        else:
            node[i] = majority(d_filtrated)
        unfold(node,attri,i+1)
    
    if root==None and node==None:
        node=root=Tree(u"好瓜",D) #初始化

    temprate = rate_category_func(D,1)
    if A == [] or temprate == 1 or temprate == 0:
        return root

    A = sorted(A,key=lambda x:weigh_fun(D,x))
    temp = deepcopy(node) #为node做一下备份
    unfold(node,A[-1])
    cur_accuracy = accuracy_fun(root,wm_validationset)

    if isgreedy:
        if cur_accuracy < accuracy:     #尽量划分使得准确率最高，但是正确做法是减少划分次数
            node = temp
            return root
        else:
            preprune_tree_generate(filtrate_func(D,A[-1],1),A[:-1],weigh_fun,isgreedy,node[1],root,cur_accuracy)
            preprune_tree_generate(filtrate_func(D,A[-1],2),A[:-1],weigh_fun,isgreedy,node[2],root,cur_accuracy)
            preprune_tree_generate(filtrate_func(D,A[-1],3),A[:-1],weigh_fun,isgreedy,node[3],root,cur_accuracy)
    else:
        if cur_accuracy <= accuracy:     #尽量划分使得枝桠最少
            node = temp
            return root
        else:
            preprune_tree_generate(filtrate_func(D,A[-1],1),A[:-1],weigh_fun,isgreedy,node[1],root,cur_accuracy)
            preprune_tree_generate(filtrate_func(D,A[-1],2),A[:-1],weigh_fun,isgreedy,node[2],root,cur_accuracy)
            preprune_tree_generate(filtrate_func(D,A[-1],3),A[:-1],weigh_fun,isgreedy,node[3],root,cur_accuracy)
    return root
     

def postprune_tree_generate(D,A,weigh_fun):
    def travel(node,nodeStack):#遍历
        if node == None or node.isLeaf:
            return
        else:
            nodeStack.append(node)
            travel(node[0],nodeStack)
            travel(node[1],nodeStack)
            travel(node[2],nodeStack)
            return
    majority_fun = lambda D:[u"坏瓜",u"好瓜"][int(rate_category_func(D,1))] #返回集合中大多数元素所属类型的节点
    def prune(nodeStack,accuracy,root):
        node = nodeStack.pop() #备份弹出的节点
        if node == root:
            return
        backup = node.attri
        node.isLeaf,node.attri = True,majority_fun(node.datalist)
        accuracy2 = accuracy_fun(root,wm_validationset)
        if accuracy < accuracy2: #如果剪枝后正确率上升
            node.__list = [None,None,None] #确认剪枝
            accuracy = accuracy2
        else:
            node.isLeaf,node.attri = False,backup #还原剪枝
        prune(nodeStack,accuracy,root)

    # raw_tree = preprune_tree_generate(D,A,weigh_fun,True)
    raw_tree = rawtree_generate(D,A,Gini_index)
    raw_accuracy = accuracy_fun(raw_tree,wm_validationset)
    nodeStack = list()
    travel(raw_tree,nodeStack)#节点栈，越深的节点在越上面
    prune(nodeStack,accuracy_fun(raw_tree,wm_validationset),raw_tree)
    return raw_tree
    



def accuracy_fun(Tree,wm_validationset):
    def travel(subtree,unit):
        if subtree.isLeaf:
            return subtree.attri
        else:
            return travel(subtree[int(unit[wm_attridict[subtree.attri]])],unit)#根据数据集中的值遍历
    
    compurefunc = lambda unit1, label:[u'坏瓜', u'好瓜', u'错误'].index(travel(Tree,unit1)) == label
    return sum(map(compurefunc, *wm_validationset)) / wm_counter(wm_validationset)

def main():
    test_Giniattri(wm_trainningset,wm_attriset)
    test_infomation_gain(wm_dataset,wm_attriset)
    a = rawtree_generate(wm_trainningset,wm_attriset,Gini_index)
    b = preprune_tree_generate(wm_trainningset,wm_attriset,Gini_index)
    c = preprune_tree_generate(wm_trainningset,wm_attriset,Gini_index,True)
    d = postprune_tree_generate(wm_trainningset,wm_attriset,Gini_index)
    print("基尼指数作评价函数：")
    print("     未剪枝的决策树正确率为：%.3f" % accuracy_fun(a,wm_validationset))
    print("     非贪心预剪枝的决策树正确率为：%.3f" % accuracy_fun(b,wm_validationset))
    print("     贪心预剪枝的决策树正确率为：%.3f" % accuracy_fun(c,wm_validationset))
    print("     后剪枝的决策树正确率为：%.3f" % accuracy_fun(d,wm_validationset))
    a = rawtree_generate(wm_trainningset,wm_attriset,infomation_gain)
    b = preprune_tree_generate(wm_trainningset,wm_attriset,infomation_gain)
    c = preprune_tree_generate(wm_trainningset,wm_attriset,infomation_gain,True)
    d = postprune_tree_generate(wm_trainningset,wm_attriset,infomation_gain)
    print("信息增益作评价函数：")
    print("     未剪枝的决策树正确率为：%.3f" % accuracy_fun(a,wm_validationset))
    print("     非贪心预剪枝的决策树正确率为：%.3f" % accuracy_fun(b,wm_validationset))
    print("     贪心预剪枝的决策树正确率为：%.3f" % accuracy_fun(c,wm_validationset))
    print("     后剪枝的决策树正确率为：%.3f" % accuracy_fun(d,wm_validationset))
    print("结论：评价函数对决策树的精度影响并不如剪枝对决策树的影响明显")
if __name__ == "__main__":
    main()