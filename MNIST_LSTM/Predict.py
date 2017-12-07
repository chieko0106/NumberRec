# -*- coding: UTF-8 -*-
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

#载入数据

father_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
data_path = father_path+'/data/'
test = pd.read_csv(data_path+'test.csv')
test_pics = np.array(test.iloc[0:,0:],dtype=np.float32)


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

# **步骤6：按时间步展开计算
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

#初始化,设置CPU按需求增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

'''
定义模型完成
'''


'''
预测
'''
#载入训练数据
saver = tf.train.Saver()
saver.restore(sess, "LSTM_train_result.ckpt")
#进行预测
predint = []
prediction=tf.argmax(y_pre,1)
all = test_pics.shape[0]
for i in range(all):
    predint.append([i+1,prediction.eval(feed_dict={_X: [test_pics[i]],keep_prob: 1.0,batch_size:1}, session=sess)[0]])
    print("predicted int is {}".format(predint[-1][-1]))

#预测结果写入文件
Out = pd.DataFrame(np.array(predint),columns=['ImageId','Label'])
Out.to_csv("test.csv",index=False,sep=',')

print('Predict finished')