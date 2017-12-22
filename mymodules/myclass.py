#encoding:utf-8
#决策树用的数据结构,支持下标访问
class Tree:
    #包含节点名，子树地址，是否是叶子节点等
    def __init__(self,attri,datalist,isLeaf = False):
        self.__attri = attri
        self.__list = [None,None,None]
        self.__datalist = datalist
        self.__isLeaf = isLeaf
    def __getitem__(self,key):
        key -= 1
        if self.__list == None:
            return None
        elif key > 3:
            return None
        return self.__list[key]
    def __setitem__(self,key,value):
        if key == 0:
            return False
        key -= 1
        self.__list[key] = value
        return True
    @property
    def childTree(self):
        return self.__list
    @property
    def isLeaf(self):
        return self.__isLeaf
    @property
    def attri(self):
        return self.__attri
    @property
    def datalist(self):
        return self.__datalist

    @attri.setter
    def attri(self,value):
        self.__attri = value
    @childTree.setter
    def childTree(self,value):
        self.__list = value
    @isLeaf.setter
    def isLeaf(self,value):
        self.__isLeaf = value

    def __curstr__(self):
        l = '['
        for x in self.__list:
            if x ==None:
                continue
            l += (x.attri + ',')
        return self.__attri + l +']'

    def __str__(self):
        def travel(node,depth = 0,L=[]):
            if node == None or node.isLeaf:
                return
            else:
                try:
                    L[depth].append('   '+node.__curstr__())
                except IndexError:
                    L.append(['   '+node.__curstr__()])
                travel(node.childTree[0],depth+1,L)
                travel(node.childTree[1],depth+1,L)
                travel(node.childTree[2],depth+1,L)
        L = []
        string = "{\n "
        travel(self, 0, L)
        for x in L:
            for y in x:
                string = string + y + " "
            string += "\n "
        return string + "}"

if __name__ == '__main__': 
    print("cannot execute this module")