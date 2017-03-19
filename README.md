# SimGAN_project

Tensorflow implementation of GAN and related works.

I tried several versions of GAN to generate MNIST data.

To see the results in detail, visit 
[sample_output directory](https://github.com/RyoMazda/SimGAN_project/tree/master/sample_output) 
or you can git clone and try them by yourself.

## GAN_naive.py

The simplest implementation of GAN to generate MNIST images.

Both Discriminator and Generator are single layer Neural Network with the number of nodes is 256.

![GAN_naive](/sample_images/GAN_naive.png)

We can tell that this is not from over-fitting because we can get a continuous map.

![GAN_naive](/sample_images/GAN_naive_2Dmap.png)


## GAN_batchnorm.py

Batch normalization is applied to GAN_naive.

![GAN_naive](/sample_images/GAN_bn.png)

Seems like this doesn't make generated images better.
It's always possible that the parameter-tuning is not enough though.


### Comments

* Batch norm to Discriminator

The weird thing is that when batch norm is applied to Discriminator, only black plain images are generated.

I guess this is because D is too strong for G at the beginning of the training and G gets stuck at a local minimum.


## GAN_separate.py

I separate the training process of D
because it's recommended by
[this (How to train a GAN)](https://github.com/soumith/ganhacks).

Before:
```math
for each batch:
    min_{\theta_D} \left( z * - \log (D(x_{real})) + (1-z) * - \log (1 - D(G(z))) \right)
```

After:
```math
for each batch:
    min_{\theta_D} \left( z * - \log (D(x_{real})) \right)
    min_{\theta_D} \left( (1-z) * - \log (1 - D(G(z))) \right)
```

I found this just makes the training process SLOWER and generated images WORSE.

![GAN_naive](/sample_images/GAN_separate.png)

It's still possible that this generates better images if trained more (with larger epochs), meaning the separation make the training slower but stable.


## GAN_double.py

Two layers version.



## DCGAN_mnist.py

Convolusion is applied.

Not succeeded yet...


## SimGAN.py

Work in progress...


## References

### Original papers

[GAN](https://arxiv.org/abs/1406.2661)

[DCGAN](https://arxiv.org/abs/1511.06434)

[SimGAN](https://arxiv.org/abs/1612.07828)


### GAN Techniques

[How to train a GAN](https://github.com/soumith/ganhacks)

[GAN techniques](https://arxiv.org/abs/1606.03498)


## My Environment

* Python 3.5.1
* Tensorflow 1.0.1
