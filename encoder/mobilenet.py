"""
The MIT License (MIT)

Original Work: Copyright (c) 2016 Ryan Dahl
(See: https://github.com/ry/tensorflow-resnet)

Modified Work: Copyright (c) 2017 Marvin Teichmann

For details see 'licenses/RESNET_LICENSE.txt'
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import tensorflow.contrib.slim as slim

from nets import mobilenet_v1

import datetime
import numpy as np
import os
import time

import logging

MOVING_AVERAGE_DECAY = 0.998
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
MOMENTUM = 0.9
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = tf.GraphKeys.UPDATE_OPS
# must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]


def inference(hypes, images, train=True, reuse=None,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              preprocess=True,
              bottleneck=True):
    # if preprocess is True, input should be RGB [0,1], otherwise BGR with mean
    # subtracted

    if preprocess:
        x = _inception_preprocess(images)

    is_train = tf.convert_to_tensor(train,
                                       dtype='bool',
                                       name='is_training')

    if reuse is None:
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_train)):
            net, end_points = mobilenet_v1.mobilenet_v1_base(x)
    		
    else:
        with tf.variable_scope("MobilenetV1", [x], reuse = reuse) as scope:
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_train)):
                net, end_points = mobilenet_v1.mobilenet_v1_base(x, scope = scope)


    logits = {}

    logits['images'] = images
    logits['fcn_in'] = end_points['Conv2d_13_pointwise']
    logits['feed2'] = end_points['Conv2d_11_pointwise']
    logits['feed4'] = end_points['Conv2d_5_pointwise']

    logits['early_feat'] = logits['feed2']
    logits['deep_feat'] = logits['fcn_in']

    if train:
        restore = tf.global_variables()
        hypes['init_function'] = _initalize_variables
        hypes['restore'] = restore

    return logits


def _initalize_variables(hypes):
    if hypes['load_pretrained']:
        logging.info("Pretrained weights are loaded.")
        logging.info("The model is fine-tuned from previous training.")
        restore = hypes['restore']
        init = tf.global_variables_initializer()
        sess = tf.get_default_session()
        sess.run(init)

        saver = tf.train.Saver(var_list=restore)

        filename = 'mobilenet_v1_1.0_224.ckpt'

        if 'TV_DIR_DATA' in os.environ:
            filename = os.path.join(os.environ['TV_DIR_DATA'], 'weights',
                                    "mobilenet", filename)
        else:
            filename = os.path.join('DATA', 'weights', "mobilenet",
                                    filename)

        # if not os.path.exists(filename):
        #     logging.error("File not found: {}".format(filename))
        #     logging.error("Please download weights from here: {}"
        #                   .format('network_url'))
        #     exit(1)

        logging.info("Loading weights from disk.")
        saver.restore(sess, filename)
    else:
        logging.info("Random initialization performed.")
        sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        sess.run(init)


def _inception_preprocess(rgb):    
    rgb = tf.div(rgb, 255.0)
    rgb = tf.subtract(rgb, 0.5)
    rgb = tf.multiply(rgb, 2.0)
    
    return rgb

