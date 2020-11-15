"""MAML model code"""
import numpy as np
import sys
import tensorflow as tf
from functools import partial

import numpy as np
import os
import random
import csv
import pickle
from DataLoader import *
## Loss utilities
def plot_acc(val_acc, name, save=False):
    import matplotlib.pyplot as plt
    title = f'{name}_Val_Acc'
    plt.plot(val_acc)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    if save:
        plt.savefig(f"{title}.png")
    plt.show()
    plt.clf()

def cross_entropy_loss(pred, label, k_shot):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=tf.stop_gradient(label)) / k_shot)

def accuracy(labels, predictions):
  return tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))


seed = 123
def conv_block(inp, cweight, bweight, bn, activation=tf.nn.relu, residual=False):
  """ Perform, conv, batch norm, nonlinearity, and max pool """
  stride, no_stride = [1,2,2,1], [1,1,1,1]

  conv_output = tf.nn.conv2d(input=inp, filters=cweight, strides=no_stride, padding='SAME') + bweight
  normed = bn(conv_output)
  normed = activation(normed)
  return normed

class ConvLayers(tf.keras.layers.Layer):
  def __init__(self, channels, dim_hidden, dim_output, img_size):
    super(ConvLayers, self).__init__()
    self.channels = channels
    self.dim_hidden = dim_hidden
    self.dim_output = dim_output
    self.img_size = img_size

    weights = {}

    dtype = tf.float32
    weight_initializer =  tf.keras.initializers.GlorotUniform()
    k = 3

    weights['conv1'] = tf.Variable(weight_initializer(shape=[k, k, self.channels, self.dim_hidden]), name='conv1', dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
    self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
    weights['conv2'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv2', dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
    self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
    weights['conv3'] = tf.Variable(weight_initializer(shape=[k, k, self.dim_hidden, self.dim_hidden]), name='conv3', dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
    self.bn3 = tf.keras.layers.BatchNormalization(name='bn3')
    weights['conv4'] = tf.Variable(weight_initializer([k, k, self.dim_hidden, self.dim_hidden]), name='conv4', dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
    self.bn4 = tf.keras.layers.BatchNormalization(name='bn4')
    weights['w5'] = tf.Variable(weight_initializer(shape=[self.dim_hidden, self.dim_output]), name='w5', dtype=dtype)
    weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
    self.conv_weights = weights

  def call(self, inp, weights):
    channels = self.channels
    inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
    hidden1 = conv_block(inp, weights['conv1'], weights['b1'], self.bn1)
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], self.bn2)
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], self.bn3)
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], self.bn4)
    hidden4 = tf.reduce_mean(input_tensor=hidden4, axis=[1, 2])
    return tf.matmul(hidden4, weights['w5']) + weights['b5']

class MAML(tf.keras.Model):

    def __init__(self, dim_input=(1,1,1), dim_output=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4, num_filters=32, k_shot=5, learn_inner_update_lr=False):
        super(MAML, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = inner_update_lr
        self.loss_func = partial(cross_entropy_loss, k_shot=k_shot)
        self.dim_hidden = num_filters
        self.channels = dim_input[2]
        self.img_size = dim_input[0]
        # self.img_size = int(np.sqrt(self.dim_input / self.channels))


        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]] * num_inner_updates
        losses_ts_post = [[]] * num_inner_updates
        accuracies_ts = [[]] * num_inner_updates

        # Define the weights - these should NOT be directly modified by the
        # inner training loop
        tf.random.set_seed(seed)
        self.conv_layers = ConvLayers(self.channels, self.dim_hidden, self.dim_output, self.img_size)
        self.forward = self.conv_layers

        self.learn_inner_update_lr = learn_inner_update_lr
        if self.learn_inner_update_lr:
            self.inner_update_lr_dict = {}
            for key in self.conv_layers.conv_weights.keys():
                self.inner_update_lr_dict[key] = [
                    tf.Variable(self.inner_update_lr, name='inner_update_lr_%s_%d' % (key, j)) for j in
                    range(num_inner_updates)]

    def call(self, inp, meta_batch_size=25, num_inner_updates=1):
        def task_inner_loop(inp, reuse=True, meta_batch_size=25, num_inner_updates=1):
            """
            Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
            Args:
              inp: a tuple (input_tr, input_ts, label_tr, label_ts), where input_tr and label_tr are the inputs and
                labels used for calculating inner loop gradients and input_ts and label_ts are the inputs and
                labels used for evaluating the model after inner updates.
                Should be shapes:
                  input_tr: [N*K, 784]
                  input_ts: [N*K, 784]
                  label_tr: [N*K, N]
                  label_ts: [N*K, N]
            Returns:
              task_output: a list of outputs, losses and accuracies at each inner update
            """
            # the inner and outer loop data
            input_tr, input_ts, label_tr, label_ts = inp

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            weights = self.conv_layers.conv_weights

            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(weights)
                out_tr = self.conv_layers(input_tr, weights)
                loss_tr = self.loss_func(out_tr, label_tr)

            label_tr = tf.reshape(label_tr, tf.shape(out_tr))
            correct_pred = tf.equal(tf.argmax(out_tr, axis=-1), tf.argmax(label_tr, axis=-1))
            correct_pred = tf.cast(correct_pred, tf.float32)
            tr_accuracy = tf.reduce_mean(correct_pred)

            for i in range(num_inner_updates):
                grads = tape.gradient(loss_tr, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                new_weights = dict()
                for key in weights:
                    new_weights[key] = weights[key] - self.inner_update_lr_dict[key] * gradients[key]
                    # new_weights[key] = weights[key] - self.self.inner_update_lr * gradients[key]

                out_ts = self.conv_layers(input_ts, new_weights)
                loss_ts = self.loss_func(out_ts, label_ts)
                label_ts = tf.reshape(label_ts, tf.shape(out_ts))

                task_outputs_ts.append(out_ts)
                task_losses_ts.append(loss_ts)

            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = out_tr, loss_tr, tr_accuracy

            #############################

            # Compute accuracies from output predictions
            task_accuracy_tr_pre = accuracy(tf.argmax(input=label_tr, axis=1),
                                            tf.argmax(input=tf.nn.softmax(task_output_tr_pre), axis=1))

            for j in range(num_inner_updates):
                task_accuracies_ts.append(accuracy(tf.argmax(input=label_ts, axis=1),
                                                   tf.argmax(input=tf.nn.softmax(task_outputs_ts[j]), axis=1)))

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre,
                           task_accuracies_ts]

            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
        unused = task_inner_loop((input_tr[0], input_ts[0], label_tr[0], label_ts[0]),
                                 False,
                                 meta_batch_size,
                                 num_inner_updates)
        out_dtype = [tf.float32, [tf.float32] * num_inner_updates, tf.float32, [tf.float32] * num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32] * num_inner_updates])
        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size,
                                          num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial,
                           elems=(input_tr, input_ts, label_tr, label_ts),
                           dtype=out_dtype,
                           parallel_iterations=meta_batch_size)

        return result





def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    gradients = outer_tape.gradient(total_losses_ts[-1], model.trainable_variables)
    optim.apply_gradients(zip(gradients, model.trainable_variables))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]

    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def meta_train_fn(model, exp_string, data_generator, data_loader,
                  n_way=5, meta_train_iterations=15000, meta_batch_size=25,
                  log=True, logdir='/tmp/data', k_shot=1, num_inner_updates=1, meta_lr=0.001):

    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 10
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5

    pre_accuracies, post_accuracies = [], []

    optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
    accuracies = []
    num_std = 0.15
    noise_std = 0.0008



    im_width, im_height, channels = model.dim_input


    for itr in range(meta_train_iterations):

        images, labels = data_generator.sample_batch(32, k_shot*2, n_way, num_std=num_std, noise_std=noise_std, shuffle=True, swap=False, h=im_height, w=im_width, shuffle_resolutoin=True)
        input_tr = images[:, :, :k_shot, :]
        input_ts = images[:, :, k_shot:, :]
        label_tr = labels[:, :, :k_shot, :]
        label_ts = labels[:, :, k_shot:, :]
        inp = (input_tr, input_ts, label_tr, label_ts)

        result = outer_train_step(inp, model, optimizer, meta_batch_size=meta_batch_size,
                                  num_inner_updates=num_inner_updates)

        if itr % SUMMARY_INTERVAL == 0:
            pre_accuracies.append(result[-2])
            post_accuracies.append(result[-1][-1])

        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration %d: pre-inner-loop train accuracy: %.5f, post-inner-loop test accuracy: %.5f' % (
            itr, np.mean(pre_accuracies), np.mean(post_accuracies))
            print(print_str)
            pre_accuracies, post_accuracies = [], []

        if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0:
            images, labels = data_loader.sample_batch('meta_val', meta_batch_size)

            input_tr = images[:, :, :k_shot, :]
            input_ts = images[:, :, k_shot:, :]
            label_tr = labels[:, :, :k_shot, :]
            label_ts = labels[:, :, k_shot:, :]

            inp = (input_tr, input_ts, label_tr, label_ts)
            result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

            print(
                'Meta-validation pre-inner-loop train accuracy: %.5f, meta-validation post-inner-loop test accuracy: %.5f' % (
                result[-2], result[-1][-1]))
            accuracies.append(result[-1][-1])
    model_file = logdir + '/' + exp_string + '/model' + str(itr)
    print("Saving to ", model_file)
    model.save_weights(model_file)
    print(accuracies)
    filename = "./maml_{}_{}_{}".format(n_way, k_shot, int(meta_lr*1000))
    np.save(filename, np.array(accuracies))

def meta_test_fn(model, data_loader, n_way=5, meta_batch_size=25, k_shot=1,
                 num_inner_updates=1):

    np.random.seed(1)
    random.seed(1)

    meta_test_accuracies = []
    NUM_META_TEST_POINTS = 500
    for _ in range(NUM_META_TEST_POINTS):

        images, labels = data_loader.sample_batch("meta_test", meta_batch_size)
        input_tr = images[:, :, :k_shot, :]
        input_ts = images[:, :, k_shot:, :]
        label_tr = labels[:, :, :k_shot, :]
        label_ts = labels[:, :, k_shot:, :]
        #############################
        inp = (input_tr, input_ts, label_tr, label_ts)
        result = outer_eval_step(inp, model, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

        meta_test_accuracies.append(result[-1][-1])

    meta_test_accuracies = np.array(meta_test_accuracies)
    means = np.mean(meta_test_accuracies)
    stds = np.std(meta_test_accuracies)
    ci95 = 1.96 * stds / np.sqrt(NUM_META_TEST_POINTS)

    print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))


def run_maml(data_generator, data_loader, n_way=5, k_shot=1, meta_batch_size=25, meta_lr=0.001,
             inner_update_lr=0.4, num_filters=32, num_inner_updates=1,
             learn_inner_update_lr=False,
             resume=False, resume_itr=0, log=True, logdir='/tmp/data',
             data_path='./omniglot_resized', meta_train=True,
             meta_train_iterations=15000, meta_train_k_shot=-1,
             meta_train_inner_update_lr=-1):
    # call data_generator and get data with k_shot*2 samples per class




    # set up MAML model
    dim_output = data_loader.dim_output
    dim_input = data_loader.dim_input
    model = MAML(dim_input,
                 dim_output,
                 num_inner_updates=num_inner_updates,
                 inner_update_lr=inner_update_lr,
                 k_shot=k_shot,
                 num_filters=num_filters,
                 learn_inner_update_lr=learn_inner_update_lr)

    if meta_train_k_shot == -1:
        meta_train_k_shot = k_shot
    if meta_train_inner_update_lr == -1:
        meta_train_inner_update_lr = inner_update_lr

    exp_string = 'cls_' + str(n_way) + '.mbs_' + str(meta_batch_size) + '.k_shot_' + str(
        meta_train_k_shot) + '.inner_numstep_' + str(num_inner_updates) + '.inner_updatelr_' + str(
        meta_train_inner_update_lr) + '.learn_inner_update_lr_' + str(learn_inner_update_lr)

    if meta_train:
        meta_train_fn(model, exp_string, data_generator, data_loader,
                      n_way, meta_train_iterations, meta_batch_size, log, logdir,
                      k_shot, num_inner_updates, meta_lr)
    else:
        meta_batch_size = 1

        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        print("Restoring model weights from ", model_file)
        model.load_weights(model_file)

        meta_test_fn(model, data_loader, n_way, meta_batch_size, k_shot, num_inner_updates)