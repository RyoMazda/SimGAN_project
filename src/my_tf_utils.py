import tensorflow as tf
import numpy as np

def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, init_const=0.1, name=None):
    initial = tf.constant(init_const, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def conv2d(x_4d, W, stride=2):
    return tf.nn.conv2d(x_4d, W, strides=[1, stride, stride, 1], padding='SAME')


def deconv2d(x_4d, W, num_row, pix_size, feature_size, stride=2):
    """
    I want to modify this so that we can change num_row=batch_size flexiblly.
    """
    return tf.nn.conv2d_transpose(x_4d, W,
                                  output_shape=[num_row, pix_size, pix_size, feature_size],
                                  strides=[1, stride, stride, 1])


def max_pool(x_4d, ksize=2, stride=1):
    return tf.nn.max_pool(x_4d,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding='SAME')



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


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def add_gaussian_noise(input_layer, std=0.05):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


'''
this is another code for minibatch discrimination taken from
http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/

def add_minibatch_discrimination(input, num_kernels=16, kernel_dim=8):
    x = linear(input, num_kernels * kernel_dim)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)

    return tf.concat(1, [input, minibatch_features])
'''


def add_minibatch_discrimination(feature, T, batch_size):
    '''
    Implementation of minibatch_discrimintation from https://arxiv.org/abs/1606.03498
    For the implementation on Tensorflow, I looked at
        https://github.com/jakekim1009/temporalprediction/blob/master/tensorflow_code/anthony_lstm/static_ops.py

    input: feature [batch_size, A=num_features], and T [A, B, C]
    output: feature_MBD [batch_size, A+B]
    '''
    _, A = feature.get_shape().as_list()
    _, B, C = T.get_shape().as_list()
    T_2D = tf.reshape(T, [A, B * C])
    M = tf.matmul(feature, T_2D)  # shape=[batch_size, B * C]
    M = tf.reshape(M, shape=[-1, B, C])

    # abs_dif[a,b,c] = || M_a,row_b - M_c,row_b ||_L1
    abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(M, 3) - \
               tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)), 2)
    c_MBD = tf.exp(-abs_dif)
    # almost done but we need to disclude the self distance, i.e. M_i - M_i
    # masked[a, b, a] == 0, masked[a, b, c (!=a)] = c_b(x_a, x_c)
    big = np.zeros((batch_size, batch_size), dtype='float32')
    big += np.eye(batch_size)
    big = tf.expand_dims(big, 1)
    mask = 1. - big

    c_masked = c_MBD * mask
    # tf.reduce_sum(masked, 2)[a, b] = sum_c c_b(x_a, x_c) = o(x_a)_b
    o_MBD = tf.reduce_sum(c_masked, 2) / (batch_size * batch_size - batch_size)
    feature_MBD = tf.concat([feature, o_MBD], axis=1)

    return feature_MBD












