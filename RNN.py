import sys
import tensorflow as tf
import random
import numpy as np


def build_data(n):
    xs = []
    ys = []
    for i in range(2000):
        k = random.uniform(1, 50)
        # x[i] = sin(k + i) (i = 0, 1, ..., n - 1)
        # y[i] = sin(k + n)
        x = [[np.sin(k + j)] for j in range(0, n)]
        y = [np.sin(k + n)]
        xs.append(x)
        ys.append(y)
    train_x = np.array(xs[0:1500])
    train_y = np.array(ys[0:1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])
    return (train_x, train_y, test_x, test_y)

# dynamic_rnn 支持变长
"""
tf.nn.dynamic_rnn(
cell, RNNCell的一个实例:BasicRNNCell,...
inputs, RNN的输入
如果time_major = False，输入为[batch_size, max_time, input_size] 否则为 [max_time, batch_size, input_size]
sequence_length = None,
shape: [batch_size, 1]，如果当前时间步的index超过该序列的实际长度时，则该时间步不进行计算，RNN的state复制上一个时间步的，同时该时间步的输出全部为零
initial_state = None,
dtype = None,
parallel_iterations = None,
swap_memory = False,
time_major = False,
scope = None
)
"""
# static_rnn 支持定长

# BasicRnnCellRNN
"""
a_t = b + W * h_t-1 + U * x_t-1
h_t = tanh(a_t)
o_t = c + V * h_t
y_t = softmax(o_t)
记忆指数级衰减
"""
# BasicLstmCell
"""
C_t = f_t * C_t-1 + i_t * ^C_t
f_t = sigmod(W_f * [h_t-1, x_t] + b_f) # 遗忘门
i_t = sigmod(W_i * [h_t-1, x_t] + b_i) # 输入门
^C_t = tanh(W_c * [h_t -1, x_t] + b_c) # 到目前为止所有信息
o_t = sigmod(W_o * [h_t-1, x_t] + b_o) # 输出门
h_t = o_t * tanh(C_t)
"""
# LSTMCell
"""
增加了 optional peep-hole connections | optional cell clipping | optional projection layer
C_t = f_t * C_t-1 + i_t * ^C_t
f_t = sigmod(W_f * [h_t-1, x_t, C_t-1] + b_f) # 遗忘门
i_t = sigmod(W_i * [h_t-1, x_t, C_t-1] + b_i) # 输入门
^C_t = tanh(W_c * [h_t -1, x_t] + b_c) # 到目前为止所有信息
o_t = sigmod(W_o * [h_t-1, x_t, C_t-1] + b_o) # 输出门
h_t = o_t * tanh(C_t)
"""
# GRUCell
"""
减少模型参数，和lstm效果差不多
h_t = f_t * h_t-1 + i_t * ^h_t
f_t = 1 - i_t # 遗忘门
i_t = sigmod(W_i * [h_t-1, x_t] + b_i) # 输入门
^h_t = tanh(W_h * [h_t -1, x_t] + b_h) * i_t # 到目前为止所有信息
"""

# LSTMCell参数
"""
__init__(
    num_units, 隐藏神经元数量
    use_peepholes=False, 是否用peephole连接
    cell_clip=None, float类型,通过该值裁剪单元状态
    initializer=None, 用于权重和投影矩阵初始化
    num_proj=None, int类型，投影矩阵输出维数
    proj_clip=None, float类型，投影值裁剪到[-proj_clip, proj_clip]
    num_unit_shards=None, 弃用
    num_proj_shards=None, 弃用
    forget_bias=1.0,
    state_is_tuple=True, 返回状态是c_state和m_state的2-tuple
    activation=None, 内部激活函数i，默认是tanh，别换成relu，会提督爆炸
    reuse=None, 
    name=None,
    dtype=None
)
"""



# RNNCell 是所有rnncell的父类
# 
def rnn(x, n_neurons, sequence_length = None):
    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu)
    cell = tf.nn.rnn_cell.GRUCell(num_units = n_neurons)
    output, state = tf.nn.dynamic_rnn(
                cell = cell,
                inputs = x,
                dtype = tf.float32,
                sequence_length = sequence_length,
            )
    return output, state

def multi_rnn(x, n_neurons, n_layers, sequence_length = None):
    #layers = [tf.nn.rnn_cell.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu) for i in range(n_layers)]
    layers = [tf.nn.rnn_cell.GRUCell(num_units = n_neurons) for i in range(n_layers)]
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
    output, state = tf.nn.dynamic_rnn(
                cell = multi_layer_cell,
                inputs = x,
                dtype = tf.float32,
                sequence_length = sequence_length
            )
    return output, state

def lstm(x, n_neurons, sequence_length = None, use_peepholes = False):
    cell = tf.nn.rnn_cell.LSTMCell(num_units = n_neurons, use_peepholes = use_peepholes)
    output, state = tf.nn.dynamic_rnn(
                cell = cell,
                inputs = x,
                dtype = tf.float32,
                sequence_length = sequence_length,
            )
    return output, state

def multi_lstm(x, n_neurons, n_layers, sequence_length = None, use_peepholes = False):
    layers = [tf.nn.rnn_cell.LSTMCell(num_units = n_neurons, use_peepholes = use_peepholes) for i in range(n_layers)]
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
    output, state = tf.nn.dynamic_rnn(
                cell = multi_layer_cell,
                inputs = x,
                dtype = tf.float32,
                sequence_length = sequence_length
            )
    return output, state

def bilstm(x, n_neurons, flag, sequence_length = None, use_peepholes = False):
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units = n_neurons, use_peepholes = use_peepholes)
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units = n_neurons, use_peepholes = use_peepholes)
    out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,cell_bw=lstm_cell_bw, inputs=X, sequence_length=sequence_length, dtype = tf.float32)
    bi_out = tf.concat(out, 2)
    if flag=='all_ht':
        return bi_out
    if flag=='first_ht':
        return bi_out[:,0,:]
    if flag=='last_ht':
        return bi_out[:,-1,:]
    if flag=='concat':
        return tf.concat((bi_out[:,0,:], bi_out[:,-1,:]), 1)

if __name__ == '__main__':
    length = 10
    vector_size = 1
    batch_size = 10
    test_size = 10

    train_x, train_y, test_x, test_y = build_data(10)

    X = tf.placeholder("float", [None, length, vector_size])
    Y = tf.placeholder("float", [None, 1])

    #output, state = rnn(X, 10)
    #output, state = multi_rnn(X, 10, 3)
    #output, state = lstm(X, 10)
    #output, state = multi_lstm(X, 10, 3)
    output = bilstm(X, 10, 'concat')

    W = tf.Variable(tf.random_normal([40, 1], stddev = 0.01))
    B = tf.Variable(tf.random_normal([1], stddev = 0.01))

    #pred_y = tf.matmul(state, W) + B
    #pred_y = tf.matmul(state[-1], W) + B
    #pred_y = tf.matmul(state.h, W) + B
    #pred_y = tf.matmul(state.c, W) + B
    #pred_y = tf.matmul((state[-1]).h, W) + B
    pred_y = tf.matmul(output, W) + B

    loss = tf.reduce_sum(tf.square(tf.subtract(Y, pred_y)))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(50):
            for end in range(batch_size, len(train_x), batch_size):
                begin = end - batch_size
                x_value = train_x[begin:end]
                y_value = train_y[begin:end]
                sess.run(train_op, feed_dict = {
                        X : x_value,
                        Y : y_value,
                    })

            test_indices = np.arange(len(test_x))
            np.random.shuffle(test_indices)
            test_indices = test_indices[:100]
            x_value = test_x[test_indices]
            y_value = test_y[test_indices]
            val_loss = sess.run(loss, feed_dict = {
                    X : x_value,
                    Y : y_value
                })
            print ('Run %s' % i, val_loss)

