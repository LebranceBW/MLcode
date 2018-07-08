# encoding:utf-8
# 参考西瓜书P69 🍉
import sys
from os import path
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) )  ))
from dataSet.watermelon_3alpha import watermelon_x as x
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
y = tuple([1]*8 + [0]*9)
mat_X = np.c_[np.matrixlib.matrix(x), np.ones(17)]
mat_Y = np.matrixlib.matrix(y).T

def XT(x): return x.T
def Beta_multiply_X(Beta): return lambda X: Beta.T * X  # β'X

def P1_posExample(_fBetaMulX):
    # p1
    return lambda x: (np.exp(_fBetaMulX(x)) / (1 + np.exp(_fBetaMulX(x)))).sum()


def sum(x, y): return x + y


def rotate(f):
    return lambda x, y: f(x.T, y)


def rotatex(f):
    return lambda x: f(x.T)


def accumulate_unit(_fP): return lambda x, y: -x * \
    (y - _fP(x))  # -xi(yi-pi(xi,β))


def L_partdif_Beta(Beta):
    return reduce(sum, map(rotate(accumulate_unit(P1_posExample(Beta_multiply_X(Beta)))), mat_X, mat_Y))

def secondorder_accumulate_unit(
    _fp): return lambda x: x * (x.T) * _fp(x) * (1 - _fp(x))

def L_secondeorder_Beta_BetaT(Beta):
    # σL²/σβ'σβ
    return reduce(sum, map(rotatex(secondorder_accumulate_unit(P1_posExample(Beta_multiply_X(Beta)))), mat_X))

def newBeta(Beta): return Beta - \
    (L_secondeorder_Beta_BetaT(Beta)**-1) * L_partdif_Beta(Beta)

def NewTown_Method(Beta, precision=1000):
    if precision < 0.0001:
        return Beta
    else:
        tempBeta = newBeta(Beta)
        return NewTown_Method(tempBeta, np.abs(Beta - tempBeta).sum())


def main():
    mat_beta = np.matrixlib.matrix([[0], [0], [0]])
    mat_beta = NewTown_Method(mat_beta)
    # print("mat_X = ", mat_X)
    # print("mat_Y = ", mat_Y)
    # print(u"β = ", mat_beta)

    ax = plt.subplots()[1]
    fig2 = plt.gcf()
    fig2.canvas.set_window_title('Logistic Regression')
    plt.title(u'Logistic Regression Model')
    plt.xlabel(u"Density")
    plt.ylabel(u"Sugar Rate")
    plt.xlim(0, 1)
    for i, item in enumerate(x):
        if(mat_Y[i] == 1):
            ax.plot(item[0], item[1], 'go')
        else:
            ax.plot(item[0], item[1], 'bo')
    y0 = (-mat_beta[2][0] / mat_beta[1][0]).sum()
    y1 = (-(mat_beta[2][0] + mat_beta[0][0]) / mat_beta[1][0]).sum()
    ax.plot([0, 1], [y0, y1])
    plt.show()


if __name__ == '__main__':
    main()
