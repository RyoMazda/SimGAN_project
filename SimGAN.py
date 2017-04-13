# coding: utf-8

import myutil
from my_tf_utils import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.io import imread
import glob
from PIL import Image


def train_refiner(x_fake, x_real, train_mode=1):

    # ---------------------------
    # Hyper parameters
    # ---------------------------

    epochs = 100000
    learning_rate = 0.0001
    beta1 = 0.5
    batch_size = 128
    K_D = 1
    K_R = 64

    diff_weight = 0.0001

    pix_size = 28
    X_dim = pix_size ** 2
    mask_size = 2

    D1_dim = 64 # tune this
    D2_dim = 128 # tune this
    D3_dim = 1

    R_dim = 64 # tune this

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

    def discriminator(x):
        x_4d = tf.reshape(x, [-1, pix_size, pix_size, 1])
        x_4d = add_gaussian_noise(x_4d)

        h_D1 = tf.nn.bias_add(conv2d(x_4d, W_D1, stride=2), b_D1)
        h_D1 = tf.nn.elu(batch_normalize(h_D1))

        h_D2 = tf.nn.bias_add(conv2d(h_D1, W_D2, stride=2), b_D2)
        h_D2 = tf.nn.elu(batch_normalize(h_D2))

        h_D2_flat = tf.reshape(h_D2, [-1, (pix_size // 4) ** 2 * D2_dim])
        D_logit = tf.nn.bias_add(tf.matmul(h_D2_flat, W_D3), b_D3)
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


    # ---------------------------
    # Refiner: x_fake -> x_refined
    # ---------------------------

    W_R0 = weight_variable([mask_size, mask_size, 1, R_dim], name="W_R0")
    b_R0 = bias_variable([R_dim], name="b_R0")
    # for ResNet1
    W_res1 = weight_variable([mask_size, mask_size, R_dim, R_dim], name="W_res1")
    b_res1 = bias_variable([R_dim], name="b_res1")
    W_res2 = weight_variable([mask_size, mask_size, R_dim, R_dim], name="W_res2")
    b_res2 = bias_variable([R_dim], name="b_res2")

    # for ResNet2
    W_res3 = weight_variable([mask_size, mask_size, R_dim, R_dim], name="W_res3")
    b_res3 = bias_variable([R_dim], name="b_res3")
    W_res4 = weight_variable([mask_size, mask_size, R_dim, R_dim], name="W_res4")
    b_res4 = bias_variable([R_dim], name="b_res4")

    W_R2 = weight_variable([mask_size, mask_size, R_dim, 1], name="W_R2")
    b_R2 = bias_variable([1], name="b_R2")

    # pack Discriminator variables
    theta_R = [W_R0, b_R0, W_R2, b_R2, W_res1, W_res2, b_res1, b_res2, W_res3, W_res4, b_res3, b_res4]

    def refiner(x):
        x_4d = tf.reshape(x, [-1, pix_size, pix_size, 1])

        # [-1, 28, 28, 1] -> [-1, 28, 28, 64]
        h_R0 = tf.nn.bias_add(conv2d(x_4d, W_R0, stride=1), b_R0)
        h_R0 = tf.nn.relu(batch_normalize(h_R0))

        # ResNet1 h_R0 -> h_R1
        h_res1 = tf.nn.bias_add(conv2d(h_R0, W_res1, stride=1), b_res1)
        h_res1 = tf.nn.relu(batch_normalize(h_res1))
        h_res2 = tf.nn.bias_add(conv2d(h_res1, W_res2, stride=1), b_res2)
        h_res2 = batch_normalize(h_res2)
        h_R1 = tf.nn.relu(h_res2 + h_R0)

        # ResNet2 h_R1 -> h_R2
        h_res3 = tf.nn.bias_add(conv2d(h_R1, W_res3, stride=1), b_res3)
        h_res3 = tf.nn.relu(batch_normalize(h_res3))
        h_res4 = tf.nn.bias_add(conv2d(h_res3, W_res4, stride=1), b_res4)
        h_res4 = batch_normalize(h_res4)
        h_R2 = tf.nn.relu(h_res4 + h_R1)
        
        # [-1, 28, 28, 64] -> [-1, 28, 28, 1]
        h_R3 = tf.nn.bias_add(conv2d(h_R2, W_R2, stride=1), b_R2)
        h_R3 = tf.nn.sigmoid(batch_normalize(h_R3))

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
        R_loss = diff_weight * R_loss_reg
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
    if train_mode != 0:
        saver.restore(sess, "model/SimGAN/2017_0413_152230_30.ckpt")
        print("Model restored.")

    # run training
    for epoch in range(epochs + 1):

        if epoch % 1 == 0:
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
            save_path = saver.save(sess, "model/SimGAN/"+ str(fm.dateinfo) + "_" + str(epoch) + ".ckpt")
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
    # preprocess_fonts("font_images/8257", number="eight")

    # load synthesized images(numbers out of fonts) with labels
    x_fake = load_fonts_as_np_array("eight")
    # y_fake = np.zeros([1000, 10], dtype='float32) # idn yet if we need this

    # load real images without labels
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_real = mnist.validation.images
    x_real = (x_real - 1/2) * 2  # from [0,1] to [-1, 1]
    # y_real = mnist.train.labels  # We pretend that we don't have this infomation

    # train refiner
    # train_refiner(x_fake, x_real, train_mode=0)
    train_refiner(x_fake, x_real, train_mode=1)



def preprocess_fonts(dir_path, number="number"):
    index = 0
    for image_path in glob.glob(dir_path + "/*.png"):
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((28, 28))
        path = "font_images/" + number + "/" + str(index) + ".png"
        img.save(path)

        index += 1


def load_fonts_as_np_array(number="number"):
    fake_images = []
    path = "font_images/" + number + "/*.png"

    for path in glob.glob(path):
        img = imread(path)
        img = (img / img.max() - 1/2) * 2  # rescale to [-1, 1]
        img = img.astype('float32')
        img = img.reshape([28 * 28])
        fake_images.append(img)

    return np.array(fake_images)


if __name__ == '__main__':
    main()
