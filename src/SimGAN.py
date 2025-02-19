# coding: utf-8

import myutil
from my_tf_utils import *

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.io import imread
import glob
from PIL import Image


def train_refiner(x_fake, x_real, train_mode=2):

    # ---------------------------
    # Hyper parameters
    # ---------------------------

    epochs = 1000
    learning_rate = 0.001
    beta1 = 0.5
    batch_size = 128
    K_D = 1
    K_R = 32
    K_init = 20

    diff_weight = 0.1

    pix_size = 28
    X_dim = pix_size ** 2
    mask_size = 2

    D1_dim = 128 # tune this
    D2_dim = 256 # tune this
    D3_dim = 1

    R_dim = 128 # tune this

    # mix real and fake images

    # ---------------------------
    # Discriminator: X -> Y
    # ---------------------------
    # first layer
    W_D1 = weight_variable([mask_size, mask_size, 1, D1_dim], name="W_D1")
    b_D1 = bias_variable([D1_dim], name="b_D1")
    # second layer
    W_D2 = weight_variable([mask_size, mask_size, D1_dim, D2_dim], name="W_D2")
    b_D2 = bias_variable([D2_dim], name="b_D2")
    # third layer
    W_D3 = weight_variable([(pix_size // 4) ** 2 * D2_dim, D3_dim], name="W_D3")
    b_D3 = bias_variable([D3_dim], name="b_D3")

    # pack Discriminator variables
    theta_D = [W_D1, b_D1, W_D2, b_D2, W_D3, b_D3]


    local_patches = [(0,0), (0,1), (1,0), (1,1)]

    def discriminator(x):
        x_4d = tf.reshape(x, [-1, pix_size, pix_size, 1])
        x_4d = add_gaussian_noise(x_4d)


        for i,(w,h) in enumerate(local_patches):
            x_patch = tf.slice(x_4d,
                        [0, h*pix_size//2, w*pix_size//2, 0],
                        [-1, pix_size//2, pix_size//2, 1])

            h_D1 = tf.nn.bias_add(conv2d(x_patch, W_D1, stride=1), b_D1)
            h_D1 = lrelu(batch_normalize(h_D1))
            h_D1 = max_pool(h_D1, ksize=2, stride=1)

            h_D2 = tf.nn.bias_add(conv2d(h_D1, W_D2, stride=2), b_D2)
            h_D2 = lrelu(batch_normalize(h_D2))
            h_D2 = max_pool(h_D2, ksize=2, stride=1)

            h_D2_flat = tf.reshape(h_D2, [-1, (pix_size // 4) ** 2 * D2_dim])
            D_logit = tf.nn.bias_add(tf.matmul(h_D2_flat, W_D3), b_D3)
            D_prob = tf.nn.sigmoid(D_logit)

            if i == 0:
                D_logits = D_logit
                D_probs = D_prob
            else:
                D_logits = tf.concat([D_logits, D_logit], 1)
                D_probs = tf.concat([D_probs, D_prob], 1)

        return D_probs, D_logits


    # ---------------------------
    # Refiner: x_fake -> x_refined
    # ---------------------------

    W_R0 = weight_variable([mask_size, mask_size, 1, R_dim], name="W_R0")
    b_R0 = bias_variable([R_dim], name="b_R0")
    # for ResNet1
    W_res1 = weight_variable([mask_size, mask_size, R_dim, R_dim], stddev=0, name="W_res1")
    b_res1 = bias_variable([R_dim], init_const=0, name="b_res1")
    W_res2 = weight_variable([mask_size, mask_size, R_dim, R_dim], stddev=0, name="W_res2")
    b_res2 = bias_variable([R_dim], init_const=0, name="b_res2")

    # for ResNet2
    W_res3 = weight_variable([mask_size, mask_size, R_dim, R_dim], stddev=0, name="W_res3")
    b_res3 = bias_variable([R_dim], init_const=0, name="b_res3")
    W_res4 = weight_variable([mask_size, mask_size, R_dim, R_dim], stddev=0, name="W_res4")
    b_res4 = bias_variable([R_dim], init_const=0, name="b_res4")

    W_R2 = weight_variable([mask_size, mask_size, R_dim, 1], name="W_R2")
    b_R2 = bias_variable([1], name="b_R2")

    # pack Discriminator variables
    theta_R = [W_R0, b_R0, W_R2, b_R2, W_res1, W_res2, b_res1, b_res2, W_res3, W_res4, b_res3, b_res4]

    def refiner(x):
        x_4d = tf.reshape(x, [-1, pix_size, pix_size, 1])

        # [-1, 28, 28, 1] -> [-1, 28, 28, 64]
        h_R0 = tf.nn.bias_add(conv2d(x_4d, W_R0, stride=1), b_R0)
        #h_R0 = tf.nn.relu(batch_normalize(h_R0))
        h_R0 = tf.nn.relu(h_R0)

        # ResNet1 h_R0 -> h_R1
        h_res1 = tf.nn.bias_add(conv2d(h_R0, W_res1, stride=1), b_res1)
        #h_res1 = tf.nn.relu(batch_normalize(h_res1))
        h_res1 = tf.nn.relu(h_res1)
        h_res2 = tf.nn.bias_add(conv2d(h_res1, W_res2, stride=1), b_res2)
        #h_res2 = batch_normalize(h_res2)
        h_R1 = tf.nn.relu(tf.add(h_res2, h_R0))

        # ResNet2 h_R1 -> h_R2
        h_res3 = tf.nn.bias_add(conv2d(h_R1, W_res3, stride=1), b_res3)
        h_res3 = tf.nn.relu(batch_normalize(h_res3))
        h_res4 = tf.nn.bias_add(conv2d(h_res3, W_res4, stride=1), b_res4)
        #h_res4 = batch_normalize(h_res4)
        h_R2 = tf.nn.relu(tf.add(h_res4, h_R1))
        
        # [-1, 28, 28, 64] -> [-1, 28, 28, 1]
        h_R3 = tf.nn.bias_add(conv2d(h_R2, W_R2, stride=1), b_R2)
        #h_R3 = tf.nn.tanh(batch_normalize(h_R3))
        h_R3 = tf.nn.tanh(h_R3)

        x_refined = tf.reshape(h_R3, [-1, X_dim])

        return x_refined


    # ---------------------------
    # Cost Function
    # ---------------------------

    X_real = tf.placeholder(tf.float32, shape=[None, X_dim])
    X_fake = tf.placeholder(tf.float32, shape=[None, X_dim])

    D_prob_real, D_logit_real = discriminator(X_real)
    D_prob_refined, D_logit_refined = discriminator(refiner(X_fake))

    # Loss for Discriminator
    # D wants to find that real images are real
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    # D wants to find that refined images are fake
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_refined, labels=tf.zeros_like(D_logit_refined)))
    D_loss = D_loss_real + D_loss_fake

    D_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
        D_loss, var_list=theta_D)

    # Loss for Refiner
    # R wants D to find that refined images are real
    R_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_logit_refined, labels=tf.ones_like(D_logit_refined)))
    # R wants to keep the annotation by suppressing differences between refined and original
    # this is L1 norm of (x_refined - x_fake)
    R_loss_reg = tf.reduce_mean(tf.abs(refiner(X_fake) - X_fake))
    if train_mode == 0:
        R_loss = R_loss_reg
    else:
        R_loss = R_loss_real + diff_weight * R_loss_reg

    R_solver = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
        R_loss, var_list=theta_R)

    # Check accuracy
    # the probability that D thinks real images are real
    D_real_acc = tf.reduce_mean(D_prob_real)
    # the probability that D thinks fake images are real
    # D wants this to be 0, while G wants this 1
    D_refined_acc = tf.reduce_mean(D_prob_refined)


    # ---------------------------
    # Training
    # ---------------------------

    # prepare directory for saving image data
    fm = myutil.FileManager('SimGAN')
    fm.mkdir()
    # record parameter values in a text file
    with open(fm.out_path+"params.txt", "w") as text_file:
        text_file.write("learning_rate: %f\ndiff_weight: %f\nbatch_size: %d\nK_R: %d\n"
                        % (learning_rate, diff_weight, batch_size, K_R))


    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if train_mode == 1:
        saver.restore(sess, "model/SimGAN/20.ckpt")
        print("Model restored.")
    if train_mode == 2:
        saver.restore(sess, "model/SimGAN/20_2017_0518_150148.ckpt")
        print("Model restored.")

    if train_mode == 0:
      epochs = K_init

    # run training
    for epoch in range(epochs + 1):

        if epoch % 2 == 0:
            np.random.shuffle(x_real)
            X_real_mb = x_real[:batch_size]
            np.random.shuffle(x_fake)
            X_fake_mb = x_fake[:batch_size]

            # print the result
            D_real_acc_val, D_refined_acc_val = sess.run(
                [D_real_acc, D_refined_acc],
                feed_dict={X_real: X_real_mb, X_fake: X_fake_mb})
            print('Epoch: %d, D_real_acc: %f, D_fake_acc: %f'
                  % (epoch, D_real_acc_val, D_refined_acc_val))
            left_gauge = '-' * int(D_refined_acc_val * 50)
            right_gauge = '-' * int((1- D_refined_acc_val) * 50)
            print('  D win!:' + left_gauge + 'X' + right_gauge + ':R win!')

            # save random 4 * 4 images to check training process
            x_fake_samples = X_fake_mb[:8, :]
            x_refined_samples = sess.run(
                refiner(x_fake_samples), feed_dict={X_fake: x_fake_samples})
            x_samples = np.concatenate((x_fake_samples, x_refined_samples), axis=0)
            x_samples = x_samples.reshape(-1, 28, 28)
            fig = myutil.plot_grid(x_samples, cmap='Greys_r')
            png_path = fm.out_path + '{}.png'
            plt.savefig(png_path.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        if train_mode == 0 and epoch % 10 == 0:
            save_path = saver.save(sess, "model/SimGAN/" + str(epoch) + "_" + str(fm.dateinfo) + ".ckpt")
            print("Model saved in file: %s" % save_path)

        # train Refiner
        for rep in range(K_R):
            np.random.shuffle(x_fake)
            X_fake_mb = x_fake[:batch_size]

            sess.run(R_solver, feed_dict={X_fake: X_fake_mb})

        # train Discriminator
        for rep in range(K_D):
            np.random.shuffle(x_real)
            X_real_mb = x_real[:batch_size]
            np.random.shuffle(x_fake)
            X_fake_mb = x_fake[:batch_size]

            sess.run(D_solver, feed_dict={X_real: X_real_mb, X_fake: X_fake_mb})


def main():

    # preprocess()

    # load synthesized images(numbers out of fonts) with labels
    x_fake =load_fake_images()

    # load real images without labels
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
    x_real = mnist.test.images
    x_real = (x_real - 1/2) * 2  # normalize. From [0,1] to [-1, 1]

    """
    modify the real images to check if refiner works
    comment out if this is not a test.
    """
    x_real = calc_gradient(x_real) # to see if this works
    #x_real[:,28*10:28*18] = -1 # hide middle row to see if the refiner works

    # train refinere
    """
    train_mode == 0:
        Set Refiner as an identical operator, as an initial condition for later

    train_mode == 1, 2:
        train Refiner so that x_fake is modified into looking more like x_real

    """
    #train_refiner(x_fake, x_real, train_mode=0)
    #train_refiner(x_fake, x_real, train_mode=1)
    train_refiner(x_fake, x_real, train_mode=2)


def calc_gradient(x):
    y = x.reshape(-1,28,28)
    y[:,1:28,1:28] = (y[:,1:28,1:28] - y[:,:27,1:28]) + (y[:,1:28,1:28] - y[:,1:28,:27])
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return y.reshape(-1, 28*28)


def preprocess():
    preprocess_fonts("../data/font_images/824F", number="0")
    preprocess_fonts("../data/font_images/8250", number="1")
    preprocess_fonts("../data/font_images/8251", number="2")
    preprocess_fonts("../data/font_images/8252", number="3")
    preprocess_fonts("../data/font_images/8253", number="4")
    preprocess_fonts("../data/font_images/8254", number="5")
    preprocess_fonts("../data/font_images/8255", number="6")
    preprocess_fonts("../data/font_images/8256", number="7")
    preprocess_fonts("../data/font_images/8257", number="8")
    preprocess_fonts("../data/font_images/8258", number="9")


def preprocess_fonts(dir_path, number="number"):
    """
    normalize the fonts to 28 * 28 pixels
    """
    index = 0
    dir_name = "../data/processed_fonts/" + number
    os.makedirs(dir_name)
    for image_path in glob.glob(dir_path + "/*.png"):
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((28, 28))
        path = dir_name + "/" + str(index) + ".png"
        img.save(path)

        index += 1


def load_fake_images():
    x_fake_0 = load_fonts_as_np_array("0")
    x_fake_1 = load_fonts_as_np_array("1")
    x_fake_2 = load_fonts_as_np_array("2")
    x_fake_3 = load_fonts_as_np_array("3")
    x_fake_4 = load_fonts_as_np_array("4")
    x_fake_5 = load_fonts_as_np_array("5")
    x_fake_6 = load_fonts_as_np_array("6")
    x_fake_7 = load_fonts_as_np_array("7")
    x_fake_8 = load_fonts_as_np_array("8")
    x_fake_9 = load_fonts_as_np_array("9")
    x_fake = np.concatenate([x_fake_0, x_fake_1, x_fake_2, x_fake_3, x_fake_4, x_fake_5, x_fake_6, x_fake_7, x_fake_8, x_fake_9], axis=0)
    return x_fake


def load_fonts_as_np_array(number="number"):
    fake_images = []
    path = "../data/processed_fonts/" + number + "/*.png"

    for path in glob.glob(path):
        img = imread(path)
        img = (img / img.max() - 1/2) * 2  # rescale to [-1, 1]
        img = img.astype('float32')
        img = img.reshape([28 * 28])
        fake_images.append(img)

    return np.array(fake_images)


if __name__ == '__main__':
    main()
