# coding: utf-8

import myutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    learning_rate = 0.0005
    beta1 = 0.5
    batch_size = 128
    epochs = 100000

    # model parameters
    Z_dim = 32
    X_dim = 28*28
    Y_dim = 1

    hidden_D = 256
    hidden_G = 256

    # ---------------------------
    # Discriminator: X -> Y
    # ---------------------------

    D_W1 = tf.Variable(tf.truncated_normal([X_dim, hidden_D]))
    D_b1 = tf.Variable(tf.zeros([hidden_D]))

    D_W2 = tf.Variable(tf.zeros([hidden_D, Y_dim]))
    D_b2 = tf.Variable(tf.zeros([Y_dim]))

    # pack variables related to Discriminator for training separately to Generator
    theta_D = [D_W1, D_W2, D_b1, D_b2]

    def discriminator(x):
        """
        fuck! If barch norm is applied to D, this doesn't work at all.
        Maybe D is so strong from the beginning that G loses his hope.
        """
        D_h1_raw = tf.nn.bias_add(tf.matmul(x, D_W1), D_b1)
        # mean_D, variance_D = tf.nn.moments(D_h1_raw, [0])
        # D_h1_bn = tf.nn.batch_normalization(D_h1_raw, mean_D, variance_D, None, None, 1e-5)
        D_h1 = tf.nn.relu(D_h1_raw)
        D_logit = tf.nn.bias_add(tf.matmul(D_h1, D_W2), D_b2)
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


    # ---------------------------
    # Generator: Z -> X
    # ---------------------------

    G_W1 = tf.Variable(tf.truncated_normal([Z_dim, hidden_G]))
    G_b1 = tf.Variable(tf.zeros([hidden_G]))

    G_W2 = tf.Variable(tf.zeros([hidden_G, X_dim]))
    G_b2 = tf.Variable(tf.zeros([X_dim]))

    # for batch normalization
    shift_G1 = tf.Variable(tf.zeros([hidden_G]))
    scale_G1 = tf.Variable(tf.ones([hidden_G]))

    # pack variables related to Generator for training separately to Discriminator
    theta_G = [G_W1, G_W2, G_b1, G_b2, shift_G1, scale_G1]

    def generator(z):
        G_h1_raw = tf.nn.bias_add(tf.matmul(z, G_W1), G_b1)
        mean, variance = tf.nn.moments(G_h1_raw, [0])
        G_h1_bn = tf.nn.batch_normalization(G_h1_raw, mean, variance, shift_G1, scale_G1, 1e-5)
        G_h1 = tf.nn.elu(G_h1_bn) # relu? elu? Doesn't matter much.
        G_logit = tf.nn.bias_add(tf.matmul(G_h1, G_W2), G_b2)
        X_fake = tf.nn.sigmoid(G_logit)

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
    fm = myutil.FileManamer('GAN_separate')
    fm.mkdir()
    # record parameter values in a text file
    with open(fm.out_path+"params.txt", "w") as text_file:
        text_file.write("Z_dim: %d\nhidden_D: %d\nhidden_G: %d\nlearning_rate: %f"
                    % (Z_dim, hidden_D, hidden_G, learning_rate))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # run training
    for epoch in range(epochs + 1):

        # train Discriminator
        X_mb, _ = mnist.train.next_batch(batch_size)
        sess.run([D_real_solver, D_real_acc], feed_dict={X: X_mb})

        Z_mb = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
        sess.run([D_fake_solver, D_fake_acc], feed_dict={Z: Z_mb})

        # train Generator
        Z_mb = np.random.uniform(-1., 1., size=[batch_size, Z_dim])
        sess.run([G_solver], feed_dict={Z: Z_mb})

        if epoch % (epochs / 100) == 0:
            # print the result
            """
            If G wins at first, the training will probably succeed.
            If D wins at first, it will fail.
            I think the reason is that G has a local minimum where all the pixels are zero (black)
            and if D is too strong from the beginning G will fall into this pitfall.
            Therefore G has to be stronger than D at first.
            """
            D_real_acc_val, D_fake_acc_val = sess.run([D_real_acc, D_fake_acc], feed_dict={X: X_mb, Z: Z_mb})
            print('Epoch: %d, D_real_acc: %f, D_fake_acc: %f'
                  % (epoch, D_real_acc_val, D_fake_acc_val))
            left_gauge = '-' * int(D_fake_acc_val * 50)
            right_gauge = '-' * int((1- D_fake_acc_val) * 50)
            print('  D win!:' + left_gauge + 'X' + right_gauge + ':G win!')

            # save random 4 * 4 images to check training process
            Z_samples = np.random.uniform(-1., 1., size=[16, Z_dim])
            X_samples = sess.run(generator(Z), feed_dict={Z: Z_samples})
            X_samples = X_samples.reshape(-1, 28, 28)
            fig = myutil.plot_grid(X_samples)
            png_path = fm.out_path + '{}.png'
            plt.savefig(png_path.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        # ---------------------------
        # draw 2D maps
        # ---------------------------
        if epoch % (epochs // 10) == 0:
            num_grid = 10
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
            X_samples = X_samples.reshape(-1, 28, 28)
            fig = myutil.plot_grid(X_samples)
            png_path = fm.out_path + '/2Dmap-{}.png'
            plt.savefig(png_path.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)



if __name__ == '__main__':
    train()
