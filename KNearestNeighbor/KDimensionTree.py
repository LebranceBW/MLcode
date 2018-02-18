#encoding:utf-8
'''
    在KNN中用到的KD树模块
'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatchs

def plot_init():
    '''
        初始化matplotlib
    '''
    ax1 = plt.subplots()[1]
    plt.title(u'KDTree')
    plt.xlabel(u"axis0")
    plt.ylabel(u"axis1")
    return ax1

class KDTree:
    '''
        一个KDTree代表一个区域，也存储着将该区域划分掉的边界与点
    '''

    def __init__(self, area, **kw):
        '''
            初始化KD节点
            area, edge, axis, left_tree right_tree dot total_dimesion
            area: ((xmin, ymin, zmin, ...), (xmax, ymax, zmax,...))
        '''
        self.__area = area
        self.__left_tree = kw.get('left_tree')
        self.__right_tree = kw.get('right_tree')
        self.__dot = kw.get('dot')
        self.__edge = kw.get('edge')
        self.__axis = kw.get('axis')
        self.__is_leaf = not(self.__left_tree or self.__right_tree)
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
    def dot(self):
        '''
            返回dot
        '''
        return self.__dot

    @property
    def edge(self):
        '''
            返回边界
        '''
        return self.__edge

    @property
    def axis(self):
        '''
            返回坐标轴
        '''
        return self.__axis

    @property
    def total_dimension(self):
        '''
            返回总维度
        '''
        return len(self.dot)

    def draw_myself(self, ax1):
        '''
            在matplotlib中绘制出KD树
        '''
        if self.total_dimension != 2:
            print("只能绘制二维的KD树")
            return
        ax1.add_patch(
            mpatchs.Rectangle((0, 0), 1, 1, edgecolor='b', facecolor='none')
        )
        plt.axis([0, 1, 0, 1])
        def travel(tree):
            '''
                输入根节点，绘制图形
            '''
            if not tree.is_leaf:
                left_area_bigger = tree.left_tree.area[1]
                right_area_smaller = tree.right_tree.area[0]
                ax1.plot(*zip(left_area_bigger, right_area_smaller), '--', color="c")
                ax1.plot(tree.dot[0][0], tree.dot[0][1], 'go' if tree.dot[1] == 1 else 'bo')
                travel(tree.left_tree)
                travel(tree.right_tree)
        travel(self)

    @staticmethod
    def generate_tree(iterable):
        '''
            静态方法，生成KD树
            输入点集(可迭代对象)，格式:
            ((x11, x12, x13 ...), y1)
            ((x21, x22, x23 ...), y2)
            ...
        '''
        dots = tuple(iterable)
        #格式检查
        total_dimension = len(dots[0][0])
        for each in dots:
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
                return KDTree(area=area)
            points = sorted(points, key=lambda item: item[0][axis])
            index = len(points)//2  #向下取整
            edge = points[index][0][axis]   #axis轴的划分边界
            '''
                将超体的对角坐标映射到超平面上得到坐标点，再根据超平米上的坐标点得到子空间对角坐标
            '''
            def replace(point, axis, val):
                '''
                    将point的axis坐标值换成val
                '''
                point = list(point)
                point[axis] = val
                return tuple(point)
            left_tree = build_tree(points[:index], (area[0], replace(area[1], axis, edge)), axis+1)
            right_tree = build_tree(points[index+1: ], (replace(area[0], axis, edge), area[1]), axis+1)
            return KDTree(area=area, left_tree=left_tree, right_tree=right_tree, dot=points[index], edge=edge, axis=axis)
        return build_tree(dots, ((0, )*total_dimension, (1, )*total_dimension))

def test_module(dots):
    '''
        输入点集(可迭代对象)，格式:
           ((x11, x12, x13 ...), y1)
            ((x21, x22, x23 ...), y2)
            ...)
        测试KD树
    '''
    tree = KDTree.generate_tree(dots)
    ax1 = plot_init()
    tree.draw_myself(ax1)
    plt.show()

if __name__ == '__main__':
    print("运行KD树模块测试")
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from dataSet.watermelon_3alpha import wm_dataSet
    test_module(wm_dataSet)
