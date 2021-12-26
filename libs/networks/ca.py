# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from libs.configs import cfgs

import tensorflow as tf
import tensorflow.contrib.slim as slim

# tf 1.5 以后引入 eager，在 tf 1.3 中无法使用
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()


def CoordAtt(x, w, h, reduction=16):
    # print('*****' * 8)
    # print(x)

    def coord_act(x):
        tmpx = tf.nn.relu6(x + 3) / 6
        x = x * tmpx
        return x

    # c = 1024
    # x_shape = x.get_shape().as_list()
    # [b, h, w, c] = x_shape
    w = w
    h = h
    # x_shape = tf.shape(x)
    # h = x_shape[1]
    # print(h)
    # w = x_shape[2]
    x_h = slim.avg_pool2d(x, kernel_size=[1, h], data_format="NHWC", stride=[1, 1])
    x_w = slim.avg_pool2d(x, kernel_size=[w, 1], data_format="NHWC", stride=[1, 1])
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    # mip = 64
    y = slim.conv2d(y, mip, (1, 1), stride=1, padding='VALID', normalizer_fn=slim.batch_norm, activation_fn=coord_act,
                    scope='ca_conv1')

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = slim.conv2d(x_h, c, (1, 1), stride=1, padding='VALID', normalizer_fn=None, activation_fn=tf.nn.sigmoid,
                      scope='ca_conv2')
    a_w = slim.conv2d(x_w, c, (1, 1), stride=1, padding='VALID', normalizer_fn=None, activation_fn=tf.nn.sigmoid,
                      scope='ca_conv3')

    out = x * a_h * a_w

    return out

# class AdaptiveAvgPool2D(Layer):
#     def __init__(self, output_size):
#         super((AdaptiveAvgPool2D, self).__init__()
#         self.output_size = np.array(output_size)
#
#     def call(self, x):
#             input_size= [x.shape[1], x.shape[2]]
#             stride = np.floor((input_size/self.output_size - 1))
#             kernel_size = x.shape[1:3]-(self.output_size -1) * stride
#             kernel_size = turtle(kernel_size)
#             out = tf.nn.avg_pool2d(x, ksize=kernel_size,stride=stride,paddling='VALID')
#             return out
#
#
#
# def ca_layer(input_x, out_dim, ratio, layer_name, is_training):
#     # senet
#     with tf.name_scope(layer_name):
#         h = input_x.shape[1]
#         w = input_x.shape[2]
#
#         avg_pool_x = AdaptiveAvgPool2D(input_x, (h, 1))
#
#         avg_pool_x_t = tf.transpose(avg_pool_x, [0, 1, 3, 2])
#
#         avg_pool_y = AdaptiveAvgPool2D(input_x, (1, w))
#
#         # Global_Average_Pooling
#         cat_1lay = tf.concat([avg_pool_x_t,avg_pool_y], 3)
#
#         conv_1x1 = slim.conv2d(cat_1lay, out_dim // ratio, [1, 1],
#                                trainable=is_training,
#                                activation_fn=tf.nn.relu,
#                                scope='attention_conv/1x1_1'
#                                )
#
#         x_cat_conv_split_h, x_cat_conv_split_w = conv_1x1.split([h, w], 3)
#
#         x_cat_conv_split_h = tf.transpose(x_cat_conv_split_h, [0, 1, 3, 2])
#
#         s_h = slim.conv2d(x_cat_conv_split_h // ratio, out_dim, [1, 1],
#                           trainable=is_training,
#                           activation_fn=tf.nn.sigmoid,
#                           scope='attention_conv/1x1_2'
#                           )
#
#         s_w = slim.conv2d(x_cat_conv_split_w // ratio, out_dim, [1, 1],
#                           trainable=is_training,
#                           activation_fn=tf.nn.sigmoid,
#                           scope='attention_conv/1x1_3'
#                           )
