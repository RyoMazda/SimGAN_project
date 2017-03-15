# coding: utf-8

import myutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x_4d, W):
    return tf.nn.conv2d(x_4d, W, strides=[1, 1, 1, 1], padding='SAME')


def deconv2d(x_4d, W, num_row, pix_size, feature_size):
    return tf.nn.conv2d_transpose(x_4d, W,
        output_shape=[num_row, pix_size, pix_size, feature_size],
        strides=[1, 2, 2, 1])


def max_pool_2x2(x_4d):
    return tf.nn.max_pool(x_4d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train():
    # load mnist data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # ---------------------------
    # Hyper parameters
    # ---------------------------

    lr = 0.0002
    beta1 = 0.5
    batch_size_sqared = 8
    batch_size = batch_size_sqared ** 2
    epochs = 10000

    # ---------------------------
    # Define Models
    # ---------------------------

    """
    X: image Data (real or fake)
    Discriminator: X -> Y
    Y: probability of input(X) being real image not synthesized (fake) image

    Z: source vector to generate a fake image
    Generator: Z -> X_fake = generator(Z)
    """

    # model parameters
    pix_size = 28
    pix_size_pooled = pix_size // 4

    Z_dim = 32
    X_dim = pix_size ** 2
    Y_dim = 1

    mask_size = 5

    hidden_D1 = 16
    hidden_D2 = 32
    hidden_D3 = 64

    hidden_G1 = 64
    hidden_G2 = 32
    hidden_G3 = 1

    """
    Discriminator: X -> Y
    """
    # first layer
    W_D1 = weight_variable([mask_size, mask_size, 1, hidden_D1])
    b_D1 = bias_variable([hidden_D1])
    # second layer
    W_D2 = weight_variable([mask_size, mask_size, hidden_D1, hidden_D2])
    b_D2 = bias_variable([hidden_D2])
    # third layer (fully-connected layer)
    W_D3 = weight_variable([pix_size_pooled * pix_size_pooled * hidden_D2, hidden_D3])
    b_D3 = bias_variable([hidden_D3])
    # forth layer (readout layer)
    W_D4 = weight_variable([hidden_D3, Y_dim])
    b_D4 = bias_variable([Y_dim])
    # pack variables related to Discriminator
    theta_D = [W_D1, b_D1, W_D2, b_D2, W_D3, b_D3, W_D4, b_D4]

    def discriminator(x):
        x_4d = tf.reshape(x, [-1, pix_size, pix_size, 1])
        h_D1 = tf.nn.elu(conv2d(x_4d, W_D1) + b_D1)
        h_D1_pooled = max_pool_2x2(h_D1)
        h_D2 = tf.nn.elu(conv2d(h_D1_pooled, W_D2) + b_D2)
        h_D2_pooled = max_pool_2x2(h_D2)
        h_D2_flat = tf.reshape(h_D2_pooled, [-1, pix_size_pooled * pix_size_pooled * hidden_D2])
        h_D3 = tf.nn.elu(tf.matmul(h_D2_flat, W_D3) + b_D3)
        D_logit = tf.matmul(h_D3, W_D4) + b_D4
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


    """
    Generator: Z -> X
    """
    # first layer
    W_conv_G1 = weight_variable([Z_dim, (pix_size_pooled ** 2) * hidden_G1])
    b_conv_G1 = bias_variable([hidden_G1])
    # second layer
    W_conv_G2 = weight_variable([mask_size, mask_size, hidden_G2, hidden_G1])
    b_conv_G2 = bias_variable([hidden_G2])
    # third layer
    W_conv_G3 = weight_variable([mask_size, mask_size, hidden_G3, hidden_G2])
    b_conv_G3 = bias_variable([hidden_G3])

    # pack variables related to Generator
    theta_G = [W_conv_G1, b_conv_G1, W_conv_G2, b_conv_G2, W_conv_G3, b_conv_G3]

    def generator(z):
        num_row = batch_size
        h_conv_G1 = tf.reshape(tf.matmul(z, W_conv_G1),
            shape=[-1, pix_size_pooled, pix_size_pooled, hidden_G1])
        h_conv_G1 = tf.nn.bias_add(h_conv_G1, b_conv_G1)
        # TODO: batch norm here
        h_conv_G1 = tf.nn.relu(h_conv_G1)

        h_conv_G2 = deconv2d(h_conv_G1, W_conv_G2, num_row, pix_size_pooled * 2, hidden_G2)
        h_conv_G2 = tf.nn.bias_add(h_conv_G2, b_conv_G2)
        # TODO: batch norm here
        h_conv_G2 = tf.nn.relu(h_conv_G2)


        h_conv_G3 = deconv2d(h_conv_G2, W_conv_G3, num_row, pix_size, hidden_G3)
        h_conv_G3 = tf.nn.bias_add(h_conv_G3, b_conv_G3)
        h_conv_G3 = tf.reshape(h_conv_G3, shape=[num_row, X_dim])
        X_generated = tf.nn.sigmoid(h_conv_G3)

        return X_generated


    # ---------------------------
    # Cost Function
    # ---------------------------

    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(generator(Z))

    """ loss function
    naively:
        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))
    for performance see below:
    """
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    # train function
    D_solver = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(G_loss, var_list=theta_G)

    # ---------------------------
    # Training
    # ---------------------------

    # prepare directory for saving image data
    fm = myutil.FileManamer('DCGAN')
    fm.mkdir()
    # record parameter values in a text file
    with open(fm.out_path + "params.txt", "w") as text_file:
        text_file.write("Z_dim: %d\n" % (Z_dim))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # run training
    for epoch in range(epochs):

        # train Discriminator
        X_mb, _ = mnist.train.next_batch(batch_size)
        Z_mb = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_mb})

        # train Generator
        Z_mb = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_mb})

        if epoch % (epochs / 100) == 0:
            # print the result
            print('Epoch: %d, D_loss: %f, G_loss: %f' % (epoch, D_loss_curr, G_loss_curr))

            # save random 4 * 4 images to check training process
            Z_samples = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
            X_samples = sess.run(generator(Z), feed_dict={Z: Z_samples})
            X_samples = X_samples[:16, :]
            X_samples = X_samples.reshape(-1, 28, 28)
            fig = myutil.plot_grid(X_samples)
            png_path = fm.out_path + '{}.png'
            plt.savefig(png_path.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        # ---------------------------
        # draw 2D maps
        # ---------------------------
        if epoch % (epochs / 10) == 0:
            num_grid = batch_size_sqared
            """
            make Z_sample with shape(num_grid*num_grid, 2)
            example: (num_grid = 3)
                list = [-1, 0, 1]
                Z_sample = [[-1, -1],[0, -1],[1, -1], ...,[0,1],[1, 1]]
            """
            list = np.linspace(-1, 1, num_grid)
            z1, z2 = np.meshgrid(list, list)
            Z_samples = np.c_[z1.ravel(), z2.ravel()]
            """
            fill Z_dim -2 columns with 0 to make Z_sample shape(num_grid*num_grid, Z_dim)
            example:
                Z_sample = [[-1, -1, 0,,...,0],...,[1,1,0,0,0,0,0,0]]
            """
            if Z_dim > 2:
                Z_samples = np.c_[Z_samples, np.zeros([Z_samples.shape[0], Z_dim - 2]) + 0.1]
            X_samples = sess.run(generator(Z), feed_dict={Z: Z_samples})
            X_samples = X_samples.reshape(-1, pix_size, pix_size)
            fig = myutil.plot_grid(X_samples)
            png_path = fm.out_path + '/2Dmap-{}.png'
            plt.savefig(png_path.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    train()
