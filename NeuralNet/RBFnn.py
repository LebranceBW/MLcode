#encoding:utf-8
'''
    P116 5.7 用RBF构建一个完成异或运算的神经网络
    约定：w为隐层与输出层的权值，β为隐层缩放系数，center为神经元中心
    后面带_i,_j表示单元，不带则表示全体🕸
'''
import numpy as np

class Neure:
    '''🕸
        隐层神经元结构体，包括w, β, 与样本中心center
    '''
    def __init__(self, beta, center, omega, index):
        self.beta = beta
        self.center = center
        self.omega = omega
        self.index = index

def radial_basis_func(neure_i, feature_j):
    '''
        RBF径向基函数，即pi = exp(-βi(x-ci)²)
    '''
    return np.exp(-neure_i.beta* np.sum(np.square(feature_j - neure_i.center)))

def predict_func(radial_basis):
    '''
        #网络输出φ, neure为全体隐层神经元的集合
        φj = Σwi * p(βi,ci,xj)
    '''
    def func(neures, feature_j):
        unit = lambda neure_i: neure_i.omega * radial_basis(neure_i, feature_j)
        return sum(map(unit, neures))
    return func

def error_func(predict_j):
    '''
        误差评价函数E
    '''
    def func(neure, feature, label):
        unit = lambda feature_j, label_j: 0.5*np.square(predict_j(neure, feature_j)-label_j)
        return sum(map(unit, feature, label))
    return func

def expression_func(radial_basis_func_j, predict_func_j):
    '''
        中间表达式 (φ(xj) - yj)p(xj, ci)
    '''
    def warpper(neure_i):
        def func(neure, feature_j, label_j):
            return (predict_func_j(neure, feature_j)-label_j)*radial_basis_func_j(neure_i, feature_j)
        return func
    return warpper

def e_derivative_of_omega_i_fuc(expression):
    '''
        计算E对wi的偏导
    '''
    def warpper(neure_i):
        def func(neure, feature, label):
            return sum(map(lambda x, y:expression(neure_i)(neure, x, y), feature, label))
        return func
    return warpper

def e_derivative_of_beta_i_fuc(expression):
    '''
        计算E对βi的偏导
    '''
    def warpper(neure_i):
        def func(neure, feature, label):
            return -sum(map(lambda x, y:expression(neure_i)(neure, x, y) * neure_i.omega * np.sum(np.square(x-neure_i.center)), feature, label))
        return func
    return warpper

def beta_gradient_descent(neure_i, neures, feature, label):
    '''
        整合过后的beta梯度方向
    '''
    return e_derivative_of_beta_i_fuc(expression_func(radial_basis_func, predict_func(radial_basis_func)))(neure_i)(neures, feature, label)

def omega_gradient_descent(neure_i, neures, feature, label):
    '''
        整合过后的w梯度方向
    '''
    return e_derivative_of_omega_i_fuc(expression_func(radial_basis_func, predict_func(radial_basis_func)))(neure_i)(neures, feature, label)

def loss_func(neures, feature, label):
    '''
        封装后的误差函数
    '''
    return error_func(predict_func(radial_basis_func))(neures, feature, label)
def neure_init(order, neure=None):
    '''
        初始化隐层,order 为隐层神经元个数
    '''
    if neure is None:
        neure = []
    elif order == 0:
        return neure
    neure.append(Neure(1, np.abs(np.random.rand(1, 2)), 1, 4-order))
    return neure_init(order-1, neure=neure)

def train(learning_rate, neures, trainning_set, trainning_count):
    '''
        训练函数
    '''
    for count in range(trainning_count):
        for var in neures:
            var.beta = var.beta - learning_rate * beta_gradient_descent(var, neures, trainning_set[0], trainning_set[1])
            var.omega = var.omega - learning_rate * omega_gradient_descent(var, neures, trainning_set[0], trainning_set[1])
        print("第%d次训练，误差为%.4f" % (count+1,sum(loss_func(neures,trainning_set[0],trainning_set[1]))))

def main():
    '''
        主函数入口
    '''
    #0 数据集定义
    trainning_set = [
        [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]]
    #1 参数设计
    order_hiddenlayer = 4
    learning_rate = 0.1
    trainning_count = 1000
    #2 网络初始化
    neures = neure_init(order_hiddenlayer)
    #3 训练
    train(learning_rate, neures,trainning_set, trainning_count)
    #4 显示输出
    for var in trainning_set[0]:
        print("%d xor %d is %f" % (var[0], var[1], predict_func(radial_basis_func)(neures, var)))






if __name__ == '__main__':
    main()
