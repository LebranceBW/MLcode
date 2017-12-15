#encoding:utf-8
#决策树用的数据结构
class Tree:
    def __init__(self):
        self.__attri = ""
        self.__list = [None,None,None]
        self.__isLeaf = False
    def __init__(self,attri,isLeaf = False):
        self.__attri = attri
        self.__list = [None,None,None]
        self.__isLeaf = isLeaf
    def __getitem__(self,key):
        key -= 1 #方便访问属性
        if self.__list == None:
            return None
        elif key > 3:
            return None
        return self.__list[key]

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
    
    def __curstr__(self):
        l = '['
        for x in self.__list:
            if x ==None:
                continue
            l += (x.attri + ',')
        return self.__attri + l +']'

    def __str__(self):
        def travel(node,depth = 0,L=[]):
            if(node == None):
                return
            elif(node.isLeaf):
                try:
                    L[depth].append('   '+node.__curstr__())
                except IndexError:
                    L.append(['   '+node.__curstr__()])
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
        string = "决策树为：{\n "
        travel(self,0,L)
        for x in L:
            for y in x:
                string = string + y + " "
            string += "\n "
        return string + "}"

if __name__ == '__main__': 
    print("cannot execute this module")