# encoding:utf-8
'''
    🕸 累积BP与标准BP
'''
import tensorflow as tf
import sys
import time
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataSet.watermelon_3 import wm_trainningset, wm_validationset
from itertools import compress

LEARNING_RATE = 0.05
HIDDEN_LAYER_ORDER = 8

'''
修改数据格式为:
(((x11, x12, x13), (x21, x22, x23)...), ((y1), (y2), (y3)...))
'''

wm_trainningset = map(lambda item:tuple((item[0], (item[1],))) ,wm_trainningset)
wm_trainningset = tuple(zip(*wm_trainningset))
wm_validationset = map(lambda item:tuple((item[0], (item[1],))) ,wm_validationset)
wm_validationset = tuple(zip(*wm_validationset))

def addlayer(prelayer, inputshape, outputshape, activate_func=None, name="layer"):
    '''
        添加中间层
    '''
    weighs = tf.Variable(tf.random_normal([inputshape, outputshape]), name=name + '_w')
    bias = tf.Variable(tf.zeros([1, outputshape]), name=name + '_b')
    x_mul_w_plus_b = tf.matmul(prelayer, weighs) + bias
    if activate_func is None:   #Python中的对象包含三要素：id、type、value。is判断通过id来判断的。==通过value来判断的。
        return x_mul_w_plus_b
    return activate_func(x_mul_w_plus_b)


def main():
    '''
        主函数
    '''
    # 1 定义变量
    feature_space = tf.placeholder("float", [None, 8])
    label_space = tf.placeholder("float", [None, 1])
    # 2 定义层
    hiddenlayer = addlayer(feature_space, 8, HIDDEN_LAYER_ORDER, tf.nn.sigmoid, "hiddenlayer")
    prediction = addlayer(hiddenlayer, HIDDEN_LAYER_ORDER, 1, tf.nn.sigmoid, "outputlayer")
    # 3 定义评价函数与optimizer
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - label_space), axis=[1]))
    # 4 训练集上的准确率
    error_rate = tf.reduce_mean(tf.abs(tf.cast(tf.greater(prediction, 0.5), "float")-label_space))
    with tf.name_scope('summaries1'):
        tf.summary.scalar("loss:", loss)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step, minval = 0, 1
        while True:
            sess.run(train_step, feed_dict={feature_space:wm_trainningset[0], label_space:wm_trainningset[1]})
            cur = sess.run(loss, feed_dict={feature_space:wm_validationset[0], label_space:wm_validationset[1]})
            if step%100 == 0:
                print("第%d次测试集误差:%.6f" % (step, cur))
            if cur < minval:
                minval = cur
                accuracy = 1 - sess.run(error_rate, feed_dict={feature_space:wm_validationset[0], label_space:wm_validationset[1]})
                count = step
            elif (cur - minval) > 0.1:
                print("累积训练最终误差%f" % minval)
                break
            step = step+1

        #打一个训练日志
        with open("./BPNN.log", mode="a") as logfile:
            logfile.writelines(time.strftime('%m-%d %Hu:%M', time.localtime(time.time()))+"\n")
            logfile.writelines(u"  学习率为{learningrate}, 隐含层维度为{order}, 最终测试集误差为{loss:.5f}, 训练了{count}次, 正确率为:{accu:.5f}\n"\
            .format(learningrate=LEARNING_RATE, order=HIDDEN_LAYER_ORDER, loss=minval, count=count, accu=accuracy))

if __name__ == '__main__':
    main()
