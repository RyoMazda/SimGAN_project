# GAN practice using MNIST

Tensorflow implementation of GAN to generate MNIST.

## GAN_naive.py

The simplest implementation of GAN to generate MNIST images.

Both Discriminator and Generator are single layer Neural Network with 256 nodes.

![GAN_naive](/images/sample_images/GAN_naive.png)

We can tell that this is not from over-fitting because we can get a continuous map.

![GAN_naive](/images/sample_images/GAN_naive_2Dmap.png)


## DCGAN_mnist.py

![DCGAN](/sample_output/DCGAN/17400.png)

not enough...

After implementing the [Batch Discrimination](https://arxiv.org/abs/1606.03498), it got better;

![DCGAN](/sample_output/DCGAN_with_batchDiscrimination/396000.png)

interporation

![DCGAN](/sample_output/DCGAN_with_batchDiscrimination/2Dmap-360000.png)


# References

## Original papers

[GAN](https://arxiv.org/abs/1406.2661)

[DCGAN](https://arxiv.org/abs/1511.06434)

## GAN Techniques

[How to train a GAN](https://github.com/soumith/ganhacks)

[GAN techniques](https://arxiv.org/abs/1606.03498)

[Batch Discrimination](https://arxiv.org/abs/1606.03498)
