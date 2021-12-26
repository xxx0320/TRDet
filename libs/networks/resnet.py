 # -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from libs.networks.attention import squeeze_excitation_layer, build_attention, build_inception, build_inception_attention
from libs.networks.ca import CoordAtt
from help_utils.tools import add_heatmap


def coord_act(x):
    tmpx = tf.nn.relu6(x + 3) / 6
    x = x * tmpx
    return x

def SCRBottleneck(inputs, is_training, batch_norm_decay=0.997,
                  batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope('bn_a'):
            conv1_a = slim.conv2d(inputs, 512, [1, 1], trainable=is_training,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  biases_regularizer=None)
            # bn1_a = tf.contrib.layers.layer_norm(conv1_a)
            # re_1_a = tf.nn.relu(bn1_a)
            with tf.variable_scope('bn_a/k1'):
                k1 = slim.conv2d(conv1_a, 512, [3, 3],
                                 padding='SAME', trainable=is_training,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 biases_regularizer=None)
                # bn_k1 = tf.contrib.layers.layer_norm(k1)
                # out_a = tf.nn.relu(bn_k1)
        # print ('=========')
        # print('re_1_a', tf.shape(re_1_a)[1])
        # print ('========')
        with tf.variable_scope('bn_b'):
            conv1_b = slim.conv2d(inputs, 512, [1, 1], trainable=is_training,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm,
                                  normalizer_params=batch_norm_params,
                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  biases_regularizer=None)
            # bn1_b = tf.contrib.layers.layer_norm(conv1_b)
            # re_1_b = tf.nn.relu(bn1_b)

            with tf.variable_scope('bn_b/k2'):
                k2_p = slim.avg_pool2d(conv1_b, [4, 4],
                                       padding='SAME', stride=4)
                k2 = slim.conv2d(k2_p, 512, [3, 3],
                                 padding='SAME', trainable=is_training,
                                 # activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 biases_regularizer=None)
                # bn_k2 = tf.contrib.layers.layer_norm(k2)
                # print (bn_k2)
            with tf.variable_scope('bn_b/k3'):
                k3 = slim.conv2d(conv1_b, 512, [3, 3], stride=1,
                                 padding='SAME', trainable=is_training,
                                 # activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 biases_regularizer=None)
                # bn_k3 = tf.contrib.layers.layer_norm(k3)
                # print ("@@@@@@")

                re_shape = tf.shape(conv1_b)
                # print (re_shape[1])
                up_k2 = tf.image.resize_bilinear(k2, (re_shape[1], re_shape[2]))
                add_k2 = tf.nn.sigmoid(tf.add(conv1_b, up_k2))
                out_k3 = tf.multiply(k3, add_k2)

            with tf.variable_scope('bn_k4'):
                k4 = slim.conv2d(out_k3, 512, [3, 3], stride=1,
                                 padding='SAME', trainable=is_training,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 normalizer_params=batch_norm_params,
                                 weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 biases_regularizer=None)
                # bn_k4 = tf.contrib.layers.layer_norm(k4)
                # out_b = tf.nn.relu(bn_k4)

        out = tf.concat(axis=3, values=[k1, k4])   # 按维数1拼接（横着拼）
        scr_out = slim.conv2d(out, 1024, [1, 1], padding='SAME',
                              # weights_initializer=cfgs.INITIALIZER,
                              weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              trainable=is_training,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=batch_norm_params,
                              biases_regularizer=None)
        # print ('########')
        # scr_out = tf.contrib.layers.layer_norm(scr_out)
        # scr_out += inputs
        # scr_out = tf.nn.relu(scr_out)   # 剩余块
        return scr_out

def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def resnet_base(img_batch, scope_name, is_training=True):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    特征融合
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101 or mobilenetv2. '
                                  'Check your network name.')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              # use stride 1 for the last conv4 layer.

              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=1)]
              # when use fpn, stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    #

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
        # print ('%%%%%%%%%%%%')
        # print('CCC2', tf.shape(C2)[1])
        # if cfgs.ADD_SCRNET:
        #     C2 = SCRBottleneck(C2, is_training=(is_training and not_freezed[0]))
    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(C3,
                                    blocks[2:3],
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
        # SE_C4 = squeeze_excitation_layer(C4, 1024, 16, 'SE_C4', is_training)

        # C4 = SE_C4 * C4
        # add_heatmap(C4, 'C4')

        if cfgs.ADD_FUSION:   # SF-NET

            # C3_ = end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)]
            # # channels = C3_.get_shape().as_list()
            # filters1 = tf.random_normal([3, 3, 512, 1024], mean=0.0, stddev=0.01)
            # C3_atrous_conv2d = tf.nn.atrous_conv2d(C3_, filters=filters1, rate=2, padding='SAME')
            # C3_shape = tf.shape(C3_atrous_conv2d)
            #
            # C2_ = end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)]
            # filters2 = tf.random_normal([3, 3, 256, 512], mean=0.0, stddev=0.01)
            # filters3 = tf.random_normal([3, 3, 512, 1024], mean=0.0, stddev=0.01)
            # C2_atrous_conv2d = tf.nn.atrous_conv2d(C2_, filters=filters2, rate=2, padding='SAME')
            # C2_atrous_conv2d = tf.nn.atrous_conv2d(C2_atrous_conv2d, filters=filters3, rate=2, padding='SAME')
            # C2_downsampling = tf.image.resize_bilinear(C2_atrous_conv2d, (C3_shape[1], C3_shape[2]))
            #
            # C4_upsampling = tf.image.resize_bilinear(C4, (C3_shape[1], C3_shape[2]))
            # C4 = C3_atrous_conv2d + C4_upsampling + C2_downsampling

            # C4 = slim.conv2d(C4,
            #                  1024, [5, 5],
            #                  trainable=is_training,
            #                  weights_initializer=cfgs.INITIALIZER,
            #                  activation_fn=None,
            #                  scope='C4_conv5x5')

            C3_shape = tf.shape(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)])
            C4 = tf.image.resize_bilinear(C4, (C3_shape[1], C3_shape[2]))

            # _C3 = slim.conv2d(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
            #                   1024, [3, 3],
            #                   trainable=is_training,
            #                   weights_initializer=cfgs.INITIALIZER,
            #                   activation_fn=tf.nn.relu,
            #                   scope='C3_conv3x3')
            #  调用了我写的
            _C3 = SCRBottleneck(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                                is_training=(is_training and not_freezed[0]))

            # #下采样到C2
            # C2_shape = tf.shape(end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)])
            # C3_ = end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)]
            # C4 = tf.image.resize_bilinear(C4, (C2_shape[1], C2_shape[2]))
            # C3_downsampling = tf.image.resize_bilinear(C3_, (C2_shape[1], C2_shape[2]))
            # 下采样到C4
            # C3_ = end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)]
            # C4_shape = tf.shape(C4)
            # C3_downsampling = tf.image.resize_bilinear(C3_, (C4_shape[1], C4_shape[2]))



            # C4 = tf.image.resize_bilinear(C4, (100, 100))
            # C3_downsampling = tf.image.resize_bilinear(C3_, (100, 100))
            # _C3 = SCRBottleneck(C3_downsampling, is_training=(is_training and not_freezed[0]))

            # _C3 = slim.conv2d(C3_downsampling,
            #                   1024, [3, 3],
            #                   trainable=is_training,
            #                   weights_initializer=cfgs.INITIALIZER,
            #                   activation_fn=tf.nn.relu,
            #                   scope='C3_conv3x3')
            # 没用上的inception
            # _C3 = build_inception(end_points_C3['resnet_v1_101/block2/unit_3/bottleneck_v1'], is_training)

            C4 += _C3

        if cfgs.ADD_ATTENTION:   # MDA-Net

            with tf.variable_scope('build_C4_attention',
                                   regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                # tf.summary.image('add_attention_before',
                #                  tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1))

                # SE_C4 = squeeze_excitation_layer(C4, 1024, 16, 'SE_C4', is_training)
                #
                # C4 = SE_C4 * C4


                add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1), 'add_attention_before')

                # C4_attention_layer = build_inception_attention(C4, is_training)

                C4_attention_layer = build_attention(C4, is_training)
                # C4_attention_layer = build_inception_attention(C4, is_training)

                C4_attention = tf.nn.softmax(C4_attention_layer)
                # C4_attention = C4_attention[:, :, :, 1]
                C4_attention = C4_attention[:, :, :, 0]
                C4_attention = tf.expand_dims(C4_attention, axis=-1)
                # tf.summary.image('C3_attention', C4_attention)
                add_heatmap(C4_attention, 'C4_attention')

                C4 = tf.multiply(C4_attention, C4)

                # a_h, a_w = ca.CoordAtt(C4)

                # C4 = C4 * a_h * a_w

                # SE_C4 = squeeze_excitation_layer(C4, 1024, 16, 'SE_C4', is_training)

                # C4 = SE_C4 * C4
                # tf.summary.image('add_attention_after', tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1))
                add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1), 'add_attention_after')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    if cfgs.ADD_ATTENTION:
        return C4, C4_attention_layer
    else:
        return C4


def restnet_head(input, is_training, scope_name):
    block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, _ = resnet_v1.resnet_v1(input,
                                    block4,
                                    global_pool=False,
                                    include_root_block=False,
                                    scope=scope_name)
        # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
        C5_flatten = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')
        # C5_flatten = tf.Print(C5_flatten, [tf.shape(C5_flatten)], summarize=10, message='C5_flatten_shape')

    # global average pooling C5 to obtain fc layers
    return C5_flatten
