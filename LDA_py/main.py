#encoding:utf-8
#线性分类器LDA
from dataSet.watermelon_dataset import watermelon_posiexam_x as positive_x
from dataSet.watermelon_dataset import watermelon_counterexample_x as counter_x
from dataSet.watermelon_dataset import watermelon_y as y
import numpy as np
from functools import reduce
from myModules.quickfunc import quick_fun
import matplotlib.pyplot as plt

mat_positive_x = np.matrixlib.matrix(positive_x)
mat_counter_x = np.matrixlib.matrix(counter_x)
def meanVector(vector):             #计算均值向量u
    return np.mean(vector,0)

def accumulate(vector):
    return reduce(lambda x,y: x+y,vector)

def x_mul_xT(x):
    return x * x.T
# def Sw():           #sw计算
    # return lambda mat_positive_x,mat_counter_x: accumulate(map(lambda unit:x_mul_xT(unit - meanVector(mat_positive_x)),mat_positive_x)) + accumulate(map(lambda unit:x_mul_xT(unit - meanVector(mat_counter_x)),mat_counter_x))

def Sw(f): #S
    def func(positive_x,counter_x):
        return f(positive_x) + f(counter_x)
    return func

def unit(mean):
    def func(x):
        return accumulate(map(lambda xrow: (xrow - mean(x)).T * (xrow - mean(x)),x))
    return func

def inverse_Sw(Sw):
    def func(positive_x,counter_x):
        return Sw(positive_x,counter_x).I
    return func

def oumiga(mean,inverse_Sw):
    def func(positive_x,counter_x):
        return inverse_Sw(positive_x,counter_x)*(mean(positive_x)-mean(counter_x)).T
    return func

def main():
    w = oumiga(meanVector,inverse_Sw(Sw(unit(meanVector))))(mat_positive_x,mat_counter_x)
    print("w = ",w)
    ax = plot_Init()
    

    posMeanPoint = meanVector(mat_positive_x).getA()[0]
    counterMeanPoint = meanVector(mat_counter_x).getA()[0]
    slope = w.T.getA()[0]#直线向量
    pos_crosspoint = crosspoint(posMeanPoint,slope)
    counter_crosspoint = crosspoint(counterMeanPoint,slope)
    
    for item in positive_x:
        ax.plot(item[0], item[1], 'go')
    for item in counter_x:
        ax.plot(item[0], item[1], 'bo')#样本点
    ax.plot(posMeanPoint[0],posMeanPoint[1],'gs'),
    ax.plot(counterMeanPoint[0],counterMeanPoint[1],'bs')
    ax.plot([0,1],[0,-slope[0]/slope[1]])#LDA线
    ax.plot([posMeanPoint[0],pos_crosspoint[0]],[posMeanPoint[1],pos_crosspoint[1]],':g')#正例样本点中心到LDA垂线
    ax.plot([counterMeanPoint[0],counter_crosspoint[0]],[counterMeanPoint[1],counter_crosspoint[1]],':b')#反例样本点中心到LDA垂线


    plt.show()

def crosspoint(meanVector,slope):
    w1,w2 = slope[0],slope[1]
    x0,y0= meanVector[0],meanVector[1]
    commonFactor = (-w1*y0 + w2*x0) / (w1**2 + w2**2)
    return [w2 * commonFactor,-w1 * commonFactor]
def plot_Init():
    ax = plt.subplots()[1]
    fig2 = plt.gcf()
    fig2.canvas.set_window_title('LDA Classify')
    plt.title(u'LDA Classify')
    plt.xlabel(u"Density")
    plt.ylabel(u"Sugar Rate")
    plt.xlim(0, 1)
    plt.ylim(-0.2,0.5)
    return ax

if __name__ == "__main__":
    main()