# models/ProtoNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [
                layers.Conv2D(
                    filters=num_filter,
                    kernel_size=3,
                    padding='SAME',
                    activation='linear'),
            ]

            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        return out


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):

    # ck: N, D
    ck = tf.reduce_mean(tf.reshape(x_latent, (num_classes, num_support, -1)), axis=1)
    # ck: 1, N, D
    ck = tf.reshape(ck, (1, num_classes, -1))
    # q_latent: N*Q, 1, D
    q_latent = tf.reshape(q_latent, (num_classes * num_queries, 1, -1))
    # distances: N*Q, N
    distance = tf.reduce_sum((ck - q_latent) ** 2, axis=-1)

    # pred: N, Q, N
    pred = tf.reshape(-distance, [num_classes, num_queries, num_classes])

    # compute cross entropy loss
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=pred))
    num_correct = tf.equal(tf.argmax(pred, -1), tf.argmax(labels_onehot, -1))
    num_correct = tf.cast(num_correct, tf.float32)
    acc = tf.reduce_mean(num_correct)
    #############################
    return ce_loss, acc