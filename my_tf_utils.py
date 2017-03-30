import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape, init_const=0.1):
    initial = tf.constant(init_const, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x_4d, W, stride=2):
    return tf.nn.conv2d(x_4d, W, strides=[1, stride, stride, 1], padding='SAME')


def deconv2d(x_4d, W, num_row, pix_size, feature_size, stride=2):
    return tf.nn.conv2d_transpose(x_4d, W,
                                  output_shape=[num_row, pix_size, pix_size, feature_size],
                                  strides=[1, stride, stride, 1])


def batch_normalize(X, eps=1e-8, shift=None, scale=None):

    if X.get_shape().ndims == 2:
        mean, variance = tf.nn.moments(X, [0])
        X = tf.nn.batch_normalization(X, mean, variance, shift, scale, eps)

    elif X.get_shape().ndims == 4:
        mean, variance = tf.nn.moments(X, [0, 1, 2])
        X = tf.nn.batch_normalization(X, mean, variance, shift, scale, eps)

    else:
        raise NotImplementedError

    return X


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
