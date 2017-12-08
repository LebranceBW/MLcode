#encoding:utf-8
#参考西瓜书P69 🍉
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
x = [
    [0.697,0.460],
    [0.774,0.376],
    [0.634,0.264],
    [0.608,0.318],
    [0.556,0.215],
    [0.403,0.237],
    [0.481,0.149],
    [0.437,0.211],
    [0.666,0.091],
    [0.243,0.267],
    [0.245,0.057],
    [0.343,0.099],
    [0.639,0.161],
    [0.657,0.198],
    [0.360,0.370],
    [0.593,0.042],
    [0.719,0.103],
]

y = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]

mat_X = np.c_[np.matrixlib.matrix(x),np.ones(17)]
mat_Y = np.matrixlib.matrix(y).T

XT = lambda x:x.T
Beta_multiply_X = lambda Beta: lambda X: Beta.T * X #β'X
# P1_posExample = lambda X : (np.exp(X)/(1 + np.exp(X))).sum() 
def P1_posExample(_fBetaMulX):
    return lambda x:(np.exp(_fBetaMulX(x))/(1 + np.exp(_fBetaMulX(x)))).sum()#p1
sum = lambda x,y:x+y
def rotate(f):
    return lambda x,y:f(x.T,y)
def rotatex(f):
    return lambda x:f(x.T)
accumulate_unit = lambda _fP:lambda x,y: -x*(y-_fP(x)) #-xi(yi-pi(xi,β))
# L_partdif_Beta = lambda Beta:reduce(sum,map(lambda x,y:accumulate_unit(x.T,y,P1_posExample(Beta_multiply_X(Beta,x.T))),mat_X,mat_Y)) #σL/σβ
def L_partdif_Beta(Beta):
    return reduce(sum,map(rotate(accumulate_unit(P1_posExample(Beta_multiply_X(Beta)))),mat_X,mat_Y))

secondorder_accumulate_unit = lambda _fp:lambda x:x*(x.T)*_fp(x)*(1-_fp(x))
# L_secondeorder_Beta_BetaT = lambda Beta:reduce(sum,map(lambda x:secondorder_accumulate_unit(x.T,P1_posExample(Beta_multiply_X(Beta,x.T))),mat_X)) #σL²/σβ'σβ
def L_secondeorder_Beta_BetaT(Beta):
    return reduce(sum,map(rotatex(secondorder_accumulate_unit(P1_posExample(Beta_multiply_X(Beta)))),mat_X)) #σL²/σβ'σβ

newBeta = lambda Beta:Beta - (L_secondeorder_Beta_BetaT(Beta)**-1)*L_partdif_Beta(Beta)
def NewTown_Method(Beta,precision = 1000):
    if precision < 0.0001:
        return Beta
    else:
        tempBeta = newBeta(Beta)
        return NewTown_Method(tempBeta,np.abs(Beta - tempBeta).sum())


def main():
    mat_beta = np.matrixlib.matrix([[0],[0],[0]])
    mat_beta = NewTown_Method(mat_beta)
    print("mat_X = ",mat_X)
    print("mat_Y = ",mat_Y)
    print(u"β = ",mat_beta)

    fig,ax = plt.subplots()
    fig2 = plt.gcf()
    fig2.canvas.set_window_title('Logistic Regression')
    plt.title(u'Logistic Regression Model')
    plt.xlabel(u"Density")
    plt.ylabel(u"Sugar Rate")
    plt.xlim(0,1)
    for i,item in enumerate(x):
        if(mat_Y[i] == 1):
            ax.plot(item[0],item[1],'go')
        else:
            ax.plot(item[0],item[1],'bo')
    y0 = (-mat_beta[2][0]/mat_beta[1][0]).sum()
    y1 = (-(mat_beta[2][0]+mat_beta[0][0])/mat_beta[1][0]).sum()
    ax.plot([0,1],[y0,y1])
    plt.show()

if __name__ == '__main__':
    main()