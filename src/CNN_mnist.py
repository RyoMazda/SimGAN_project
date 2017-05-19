import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x_4d, W):
    return tf.nn.conv2d(x_4d, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x_4d):
    return tf.nn.max_pool(x_4d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():

    epochs = 20000
    print_rate = 100
    batch_size = 64
    lr = 1e-4

    mask_size = 5

    n = 28
    n_pooled = 28 // 4
    Di = n * n
    D1 = 32
    D2 = 64
    D3 = 1024
    Df = 10

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, Di])
    x_4d = tf.reshape(x, [-1, n, n, 1])

    # first layer
    W_conv1 = weight_variable([mask_size, mask_size, 1, D1])
    b_conv1 = bias_variable([D1])
    h_conv1 = tf.nn.elu(conv2d(x_4d, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    W_conv2 = weight_variable([mask_size, mask_size, D1, D2])
    b_conv2 = bias_variable([D2])
    h_conv2 = tf.nn.elu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fully-connected layer
    W_fc1 = weight_variable([n_pooled * n_pooled * D2, D3])
    b_fc1 = bias_variable([D3])
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_pooled * n_pooled * D2])
    h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([D3, Df])
    b_fc2 = bias_variable([Df])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train and evaluate
    y_ = tf.placeholder(tf.float32, shape=[None, Df])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # run session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch = mnist.train.next_batch(batch_size)
        if epoch % print_rate == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (epoch, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()

