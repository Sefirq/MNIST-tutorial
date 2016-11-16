from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def print_class(oneHotLabels):
    return [i for i, x in enumerate(oneHotLabels) if x == 1][0]

def visualize(image):
    plt.imshow(out, cmap='Greys_r')
    plt.show()


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
out = 1 - mnist.train.images[5]
out = np.reshape(out, (28, 28)) # function = Wx+b
x = tf.placeholder(tf.float32, [None, 784])  # there will be the input
W = tf.Variable(tf.zeros([784, 10]))  # those can be edited by computations
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#print(print_class(mnist.train.labels[5]))


