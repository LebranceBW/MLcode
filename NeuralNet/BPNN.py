#encoding:utf-8
import tensorflow as tf
import sys
from os import path 
import numpy as np
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_3 import wm_dataset,wm_trainningset,wm_validationset

#添加隐藏层
def addlayer(x,insize,outsize,activate_func = None,name = "layer"):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal([insize,outsize]),name=name+'_w')
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([1,outsize]),name=name+'_b')
        x_mul_w_plus_b = tf.matmul(x,w) + b
        if activate_func == None:
            return x_mul_w_plus_b
        else: 
            return activate_func(x_mul_w_plus_b)
#1 定义变量
x = tf.placeholder("float",[None,8])
y_ = tf.placeholder("float",[None,1])
#2 定义层
hiddenlayer = addlayer(x,8,12,tf.nn.sigmoid,"hiddenlayer")
y = addlayer(hiddenlayer,12,1,tf.nn.sigmoid,"outputlayer")
#3 定义评价函数与optimizer
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_),axis=[1]))
with tf.name_scope('summaries1'):    
    tf.summary.scalar("loss:",loss)
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test_writer = tf.summary.FileWriter(r"G:\VSCodeWorkspace\ML_coding\NeuralNet\logs",sess.graph)
    for i in range(2000):
        sess.run(train_step,feed_dict={x:wm_trainningset[0],y_:wm_trainningset[1]})
        result = sess.run(merged,feed_dict={x:wm_trainningset[0],y_:wm_trainningset[1]})
        test_writer.add_summary(result,i) #result是summary类型的，需要放入writer中，i步数（x轴）
        if i%100 == 0:
            print("第%d次训练集误差:%.6f" % (i,sess.run(loss,feed_dict={x:wm_trainningset[0],y_:wm_trainningset[1]})))
    print("训练完成后各个网络参数为")
    for var in tf.global_variables():
        print("%s \n %s" % (var.name,sess.run(var,feed_dict={x:wm_trainningset[0],y_:wm_trainningset[1]})))
        
