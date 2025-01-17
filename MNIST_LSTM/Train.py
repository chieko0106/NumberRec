# -*- coding: UTF-8 -*-


import numpy as np
import os
import pandas as pd


'''
#载入数据
'''

father_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
data_path = father_path+'/data/'
train = pd.read_csv(data_path+'train.csv')
test = pd.read_csv(data_path+'test.csv')
pics = np.array(train.iloc[:,1:],dtype=np.float32)
labels = np.array(train.iloc[:,0])

#数据标签one-hot encode
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
labels_onehot = enc.fit_transform(labels.reshape(-1,1)).toarray() #因为之后需要softmax,所以先把数据标签转为0和1组成的数组表示

# 分割测试集
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(pics,labels_onehot,test_size=0.1,random_state=0)


import tensorflow as tf
from tensorflow.contrib import rnn

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
'''
设置超参数
'''
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
# 在 1.0 版本以后请使用 ：
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])# 注意类型必须为 tf.int32

# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 28
# 每个隐含层的节点数
hidden_size = 256
# LSTM layer 的层数
layer_num = 2
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 10

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])

'''
搭建 LSTM 模型
'''
# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
####################################################################
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 28, 28])
def lstm_cell():
    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    drop_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return(drop_cell)

# **步骤4：调用 MultiRNNCell 来实现多层 LSTM
mlstm_cell = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

# **步骤5：用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size],
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

# *************** 步骤6***************
#  RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
# **步骤6：方法二，按时间步展开计算
outputs = []
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1]


# 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，要分类的话，还需要接一个 softmax 层
# 首先定义 softmax 的连接权重矩阵和偏置
# out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
# out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)


# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.global_variables_initializer())

'''
训练
'''
train_size = train_x.shape[0]
for i in range(4000):
    _batch_size = 128
    start = i*_batch_size % train_size
    end = (i+1)*_batch_size % train_size
    if start > end:
        start = train_size-_batch_size-1
        end = -1
    #print("Strat {},End {}".format(start,end))
    batch_x = train_x[start:end]
    batch_y = train_y[start:end]
    if (i+1)%200 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            _X:batch_x, y: batch_y, keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print ("Iter{},  training accuracy {}" .format ( i,  train_accuracy))
    sess.run(train_op, feed_dict={_X:batch_x, y: batch_y, keep_prob: 0.5, batch_size: _batch_size})

# 计算测试数据的准确率
print ("test accuracy {}".format(sess.run(accuracy, feed_dict={
    _X: test_x, y: test_y, keep_prob: 1.0, batch_size:test_x.shape[0]})))




print('LSTM training finished')
saver = tf.train.Saver()
saver.save(sess, 'LSTM_train_result.ckpt')