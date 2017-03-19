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
    return tf.nn.conv2d(x_4d, W, strides=[1, 2, 2, 1], padding='SAME')


def deconv2d(x_4d, W, num_row, pix_size, feature_size):
    return tf.nn.conv2d_transpose(x_4d, W,
        output_shape=[num_row, pix_size, pix_size, feature_size],
        strides=[1, 2, 2, 1])


# No pooling for DCGAN
# def max_pool_2x2(x_4d):
#     return tf.nn.max_pool(x_4d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train():
    """
    X: image Data (real or fake)
    Discriminator: X -> Y
    Y: probability of input(X) being real image not synthesized (fake) image

    Z: source vector to generate a fake image
    Generator: Z -> X_fake = generator(Z)
    """

    # load mnist data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # ---------------------------
    # Hyper parameters
    # ---------------------------

    learning_rate = 0.0001
    beta1 = 0.5
    batch_size_squared = 8
    batch_size = batch_size_squared ** 2
    epochs = 10000

    # model parameters
    pix_size = 28

    Z_dim = 32
    X_dim = pix_size ** 2

    mask_size = 4

    hidden_D1 = 128
    hidden_D2 = 256
    hidden_D3 = 1 # this has to be 1

    hidden_G1 = 128
    hidden_G2 = 256
    hidden_G3 = 1 # this has to be 1

    # ---------------------------
    # Discriminator: X -> Y
    # ---------------------------
    # first layer
    W_D1 = weight_variable([mask_size, mask_size, 1, hidden_D1])
    b_D1 = bias_variable([hidden_D1])
    # second layer
    W_D2 = weight_variable([mask_size, mask_size, hidden_D1, hidden_D2])
    b_D2 = bias_variable([hidden_D2])
    # third layer
    W_D3 = weight_variable([(pix_size // 4) ** 2 * hidden_D2, hidden_D3])
    b_D3 = bias_variable([hidden_D3])

    # pack Discriminator variables
    theta_D = [W_D1, b_D1, W_D2, b_D2, W_D3, b_D3]

    def discriminator(x):
        x_4d = tf.reshape(x, [-1, pix_size, pix_size, 1])

        h_D1 = tf.nn.bias_add(conv2d(x_4d, W_D1), b_D1)
        mean, variance = tf.nn.moments(h_D1, [0, 1, 2])
        h_D1 = tf.nn.batch_normalization(h_D1, mean, variance, None, None, 1e-5)
        h_D1 = tf.nn.elu(h_D1)

        h_D2 = tf.nn.bias_add(conv2d(h_D1, W_D2), b_D2)
        mean, variance = tf.nn.moments(h_D2, [0, 1, 2])
        h_D2 = tf.nn.batch_normalization(h_D2, mean, variance, None, None, 1e-5)
        h_D2 = tf.nn.elu(h_D2)

        h_D2_flat = tf.reshape(h_D2, [-1, (pix_size // 4) ** 2 * hidden_D2])
        D_logit = tf.nn.bias_add(tf.matmul(h_D2_flat, W_D3), b_D3)
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


    # ---------------------------
    # Generator: Z -> X
    # ---------------------------
    # first layer
    W_conv_G1 = weight_variable([Z_dim, (pix_size // 4) * (pix_size // 4) * hidden_G1])
    b_conv_G1 = bias_variable([hidden_G1])
    # second layer
    W_conv_G2 = weight_variable([mask_size, mask_size, hidden_G2, hidden_G1])
    b_conv_G2 = bias_variable([hidden_G2])
    # third layer
    W_conv_G3 = weight_variable([mask_size, mask_size, hidden_G3, hidden_G2])
    b_conv_G3 = bias_variable([hidden_G3])

    # pack Generator variables
    theta_G = [W_conv_G1, b_conv_G1, W_conv_G2, b_conv_G2, W_conv_G3, b_conv_G3]

    def generator(z):
        num_row = batch_size

        h_conv_G1 = tf.reshape(tf.matmul(z, W_conv_G1),
            shape=[-1, pix_size // 4, pix_size // 4, hidden_G1])
        h_conv_G1 = tf.nn.bias_add(h_conv_G1, b_conv_G1)
        mean, variance = tf.nn.moments(h_conv_G1, [0, 1, 2])
        h_conv_G1 = tf.nn.batch_normalization(h_conv_G1, mean, variance, None, None, 1e-5)
        h_conv_G1 = tf.nn.relu(h_conv_G1)

        h_conv_G2 = deconv2d(h_conv_G1, W_conv_G2, num_row, pix_size // 2, hidden_G2)
        h_conv_G2 = tf.nn.bias_add(h_conv_G2, b_conv_G2)
        mean, variance = tf.nn.moments(h_conv_G2, [0, 1, 2])
        h_conv_G2 = tf.nn.batch_normalization(h_conv_G2, mean, variance, None, None, 1e-5)
        h_conv_G2 = tf.nn.relu(h_conv_G2)

        h_conv_G3 = deconv2d(h_conv_G2, W_conv_G3, num_row, pix_size, hidden_G3)
        h_conv_G3 = tf.nn.bias_add(h_conv_G3, b_conv_G3)
        h_conv_G3 = tf.reshape(h_conv_G3, shape=[num_row, X_dim])
        X_fake = tf.nn.sigmoid(h_conv_G3)

        return X_fake


    # ---------------------------
    # Cost Function
    # ---------------------------

    X = tf.placeholder(tf.float32, shape=[None, X_dim])
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    D_prob_real, D_logit_real = discriminator(X)
    D_prob_fake, D_logit_fake = discriminator(generator(Z))

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    # train function
    D_real_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(D_loss_real, var_list=theta_D)
    D_fake_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(D_loss_fake, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(G_loss, var_list=theta_G)

    # Check accuracy
    # the probability that D thinks real images are real
    D_real_acc = tf.reduce_mean(D_prob_real)
    # the probability that D thinks fake images are real
    # D wants this to be 0, while G wants this 1
    D_fake_acc = tf.reduce_mean(D_prob_fake)

    # ---------------------------
    # Training
    # ---------------------------

    # prepare directory for saving image data
    fm = myutil.FileManamer('DCGAN')
    fm.mkdir()
    # record parameter values in a text file
    with open(fm.out_path + "params.txt", "w") as text_file:
        text_file.write("learning_rate: %f\nZ_dim: %d\nhidden_G1: %d\nhidden_G2: %d\nhidden_G3: %d\n"
                        % (learning_rate, Z_dim, hidden_G1, hidden_G2, hidden_G3))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # run training
    for epoch in range(epochs):

        # train Discriminator
        X_mb, _ = mnist.train.next_batch(batch_size)
        sess.run([D_real_solver], feed_dict={X: X_mb})

        Z_mb = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
        sess.run([D_fake_solver], feed_dict={Z: Z_mb})

        # train Generator
        Z_mb = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
        sess.run([G_solver], feed_dict={Z: Z_mb})

        if epoch % (epochs / 100) == 0:
            # print the result
            D_real_acc_val, D_fake_acc_val = sess.run([D_real_acc, D_fake_acc], feed_dict={X: X_mb, Z: Z_mb})
            print('Epoch: %d, D_real_acc: %f, D_fake_acc: %f'
                  % (epoch, D_real_acc_val, D_fake_acc_val))
            left_gauge = '-' * int(D_fake_acc_val * 50)
            right_gauge = '-' * int((1- D_fake_acc_val) * 50)
            print('  D win!:' + left_gauge + 'X' + right_gauge + ':G win!')

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
            num_grid = batch_size_squared
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
