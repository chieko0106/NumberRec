# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np


#载入数据

father_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
data_path = father_path+'/data/'
test = pd.read_csv(data_path+'test.csv')
test_pics = np.array(test.iloc[0:,0:],dtype=np.float32)


'''
定义模型
'''

#建立神经网络
import tensorflow as tf
sess = tf.InteractiveSession()

#定义占位符x和y_,x为图片信息,y为图对应的文字
x = tf.placeholder("float",shape=[None,784]) #张量空间x的None方向的纬度是没有被指定的,下同
y_ = tf.placeholder("float",shape=[None,10])

W = tf.Variable(tf.zeros((784,10)))# W 和 b 是待学习的量,其初始值可以是任意的.设置初始值为0.
b = tf.Variable(tf.zeros((10,)))
y = tf.nn.softmax(tf.matmul(x, W) + b)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001) #标准差stddev = 0.0001的时候收敛有点慢,所以扩大了一点点
    return tf.Variable(initial) #返回一个shape形状的正态分布初始值

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial) #返回一个shape形状的全是0.1的初始变量

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#卷积函数 指定步长为1,边缘处直接复制过来

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #定义2x2最大值池化函数

#因为要进行卷积操作，所以要将图片reshape成28*28*1的形状。
x_image = tf.reshape(x,[-1,28,28,1])


#第一层卷积层 + relu正则函数 + 池化层
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #池化

#第二层卷积层 + relu正则函数 + 池化层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#第一个全连接层 + relu正则函数 + 随机失活
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#第二个全连接层 + softmax输出
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#设置优化方法
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #交叉熵损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使交叉熵最小的优化器
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

'''
定义模型完成
'''
#初始化所有的变量
sess.run(tf.global_variables_initializer())
#载入训练数据
saver = tf.train.Saver()
saver.restore(sess, "train_result.ckpt")
#进行预测
predint = []
prediction=tf.argmax(y_conv,1)
all = test_pics.shape[0]
for i in range(all):
    predint.append([i+1,prediction.eval(feed_dict={x: [test_pics[i]],keep_prob: 1.0}, session=sess)[0]])
    print("predicted int is {}".format(predint[-1][-1]))

#预测结果写入文件
Out = pd.DataFrame(np.array(predint),columns=['ImageId','Label'])
Out.to_csv("test.csv",index=False,sep=',')

print('Predict finished')