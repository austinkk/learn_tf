
# coding: utf-8

# In[1]:





# In[2]:


import tensorflow as tf


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


import os


# In[5]:


import time


# In[6]:


import numpy as np


# In[7]:


INPUT_NODE =  784


# In[8]:


OUTPUT_NODE = 10


# In[9]:


IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10


# In[10]:


CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5


# In[11]:


FC_SIZE = 512


# In[12]:


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], 
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
    
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], 
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
    
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [-1, nodes])
    
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_bias = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
    
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
    
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_bias)
    
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [ FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))        
        
        logit = tf.matmul(fc1, fc2_weights) + fc2_bias
    return logit


# In[13]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99

REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


# In[14]:


MODEL_SAVE_PATH = "/Users/huangyukun/workspace/learn_tf/model/"
MODEL_NAME = "model.ckpt"


# In[15]:


def train(mnist):
    x = tf.placeholder(tf.float32,[None, INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])
    reshaped_x = tf.reshape(x,  [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    y = inference(reshaped_x, True,regularizer)
    global_step = tf.Variable(0, trainable = False)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #cross_entropy_mean = tf.reduce_mean(cross_entropy)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, acc = sess.run([train_op, loss, global_step, accuracy], feed_dict = {x: xs, y_: ys})
            if i % 1000 == 0:
                print "after %d steps,  loss on training batch is %g, accuracy is %g" % (step, loss_value, acc)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)


# In[16]:


mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


# In[17]:


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [mnist.validation.num_examples, INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [mnist.validation.num_examples, NUM_LABELS])
    reshaped_x = tf.reshape(x,  [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    y = inference(reshaped_x, False, None)

    corrrect_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(corrrect_prediction, tf.float32))
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    
    xs = mnist.validation.images
    ys = mnist.validation.labels
    #xs = xs.reshape((mnist.validation.num_examples, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    if True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            #print ckpt.model_checkpoint_path
            if ckpt and ckpt.model_checkpoint_path:
                for path in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess, path)
                    global_step = path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,  feed_dict = {x:xs, y_:ys})
                    print "after %s steps,  accuracy on valid is %g" % (global_step, accuracy_score)
            else:
                return


# In[18]:


#train(mnist)
evaluate(mnist)

