#encoding:utf-8
'''
    KD树模块
'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatchs

def plot_init():
    '''
        初始化
    '''
    ax1 = plt.subplots()[1]
    # fig2 = plt.gcf()
    # fig2.canvas.set_window_title('LDA Classify')
    plt.title(u'KDTree')
    plt.xlabel(u"axis0")
    plt.ylabel(u"axis1")
    return ax1

class KDTree():
    '''
        KD树类
        left_tree 左子树（小于）
        right_tree 右子树（大于）
        is_leaf 是否是叶子节点
        area 体现为超立方体对角坐标，且前一个的每一个轴的值总是比后一个要小
        格式：[(x1, y1, z1 ...), (x2, y2, z2....)]
        total_dimesions 总维度
        pointsset 点集
        (<feature>, label)
    '''

    def __init__(self, pointsset=None, area=None, is_leaf=False, left_tree=None, right_tree=None, total_dimension=0):
        '''
            初始化KD节点

        '''
        self.__is_leaf = is_leaf
        self.__area = area
        self.__left_tree = left_tree
        self.__right_tree = right_tree
        self.__pointsset = pointsset
        self.__total_dimension = total_dimension

    #又臭又长的属性定义
    @property
    def is_leaf(self):
        '''
            返回is_leaf
        '''
        return self.__is_leaf

    @property
    def area(self):
        '''
            返回area
        '''
        return self.__area

    @property
    def left_tree(self):
        '''
            返回left_tree
        '''
        return self.__left_tree

    @property
    def right_tree(self):
        '''
            返回right_tree
        '''
        return self.__right_tree

    @property
    def pointsset(self):
        '''
            返回right_tree
        '''
        return self.__pointsset

    def draw_myself(self):
        '''
            在matplotlib中绘制出KD树
        '''
        if self.__total_dimension != 2:
            print("只能绘制二维的KD树")
            return
        ax1 = plot_init()
        ax1.add_patch(
            mpatchs.Rectangle((0, 0), 1, 1, edgecolor='b', facecolor='none')
        )
        plt.axis([0, 1, 0, 1])
        def travel(tree):
            '''
                输入根节点，绘制图形
            '''
            if len(tree.pointsset) > 1:
                left_area_bigger = tree.left_tree.area[1]
                right_area_smaller = tree.right_tree.area[0]
                ax1.plot(*zip(left_area_bigger, right_area_smaller))
                travel(tree.left_tree)
                travel(tree.right_tree)
        travel(self)
        for var in self.__pointsset:
            ax1.plot(var[0][0], var[0][1], 'go' if var[1] == 1 else 'bo')
        plt.show()

    @staticmethod
    def generate_tree(iterable):
        '''
            静态方法，生成KD树
            输入点集(可迭代对象)，格式:
            ((x11, x12, x13 ...), y1)
            ((x21, x22, x23 ...), y2)
            ...
        '''
        pointsset = tuple(iterable)
        #格式检查
        total_dimension = len(pointsset[0][0])
        for each in pointsset:
            if not isinstance(each, (tuple, list)):
                raise RuntimeError("生成KD树的数据不合法")
            elif len(each) != total_dimension:
                raise RuntimeError("数据维度不统一")

        #递归建立
        def build_tree(points, area, axis=0):
            '''
                输入点集与划分轴以及总区域
                ponits:
                ((x11, x12, x13 ...), y1)
                ((x21, x22, x23 ...), y2) ...
                area:
                (x0, y0, z0) -> (x1, y1, z1)
            '''
            if axis == total_dimension:
                axis = 0

            if not points:
                '''
                    划分完成后直接返回一个最小区域
                '''
                return KDTree(pointsset=points, is_leaf=True, total_dimension=total_dimension, area=area)
            points = sorted(points, key=lambda item: item[0][axis])
            index = len(points)//2#向下取整

            diagonal_points = [list(area[0]), list(area[1])]
            for each in diagonal_points:
                each[axis] = points[index][0][axis]
            '''
                将超体的对角坐标映射到超平面上得到坐标点，再根据超平米上的坐标点得到子空间对角坐标
            '''
            left_tree = build_tree(points[:index], (area[0], tuple(diagonal_points[1])), axis+1)
            right_tree = build_tree(points[index+1: ], (tuple(diagonal_points[0]), area[1]), axis+1)
            return KDTree(pointsset=points, area=area, left_tree=left_tree, right_tree=right_tree, total_dimension=total_dimension)
        return build_tree(pointsset, ((0, )*total_dimension, (1, )*total_dimension))

def test_module(pointsset):
    '''
        输入点集(可迭代对象)，格式:
           ((x11, x12, x13 ...), y1)
            ((x21, x22, x23 ...), y2)
            ...)
        测试KD树
    '''
    tree = KDTree.generate_tree(pointsset)
    tree.draw_myself()

if __name__ == '__main__':
    print("运行模块测试")
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from dataSet.watermelon_3alpha import wm_dataSet
    test_module(wm_dataSet)
