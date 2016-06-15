import os
import os.path
import sklearn.cross_validation
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf


#define NN
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def CNN():
    W_conv1 = weight_variable([5, 5, 3, 16])
    b_conv1 = bias_variable([16])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #softmax
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())


training_X = np.load("training.X.npy")
training_y = np.load("training.y.npy")
n_training = len(training_y)

n1 = 288*384
nclasses = 2

#print n1,nclasses

x = tf.placeholder(tf.float32, [None, n1])
W = weight_variable([n1, nclasses])
b = bias_variable([nclasses])
y = tf.sigmoid(tf.matmul(x, W) + b) #tf.nn.softmax(tf.matmul(x, W) + b)
#py = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, nclasses])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy = tf.reduce_sum(- y * tf.log(y_), 1)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#now train
init = tf.initialize_all_variables()

print "starting training...",n_training

with tf.Session() as s:
    s.run(init)

    for i in xrange(1000):
        batch_x = np.empty((1,n1))
        batch_y = np.zeros((1,2))
        for k in xrange(n_training):
            image = training_X[k,:,:]
            image = image.flatten()

            
            batch_x[0,:] = image
            if training_y[k] == 1:
                batch_y[0,0] = 1
                batch_y[0,1] = 0
            else:
                batch_y[0,0] = 0
                batch_y[0,1] = 1
            
            s.run(train_step, feed_dict={x: batch_x, y_: batch_y})    

            #print sess.run(W)
            #print sess.run(b)
            #print sess.run(cross_entropy,feed_dict={x: batch_xs, y_: batch_ys})    
        w = s.run(W,feed_dict={x: batch_x, y_: batch_y})
        print('step {0}, training W {1}'.format(i, w))
        train_accuracy = s.run(accuracy,feed_dict={x: batch_x, y_: batch_y})
        print('step {0}, training accuracy {1}'.format(i, train_accuracy))
