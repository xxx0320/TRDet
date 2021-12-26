# coding:utf-8
import tensorflow as tf
import os


ckpt_path = '/Data/pbc/R2CNN-Plus-Plus_h/output/trained_weights/R2CNN_210325_pbcdata_augv1'
meta_path = os.path.join(ckpt_path, 'voc_300000model.ckpt')

saver = tf.train.import_meta_graph(meta_path + '.meta', clear_devices=True)
graph = tf.get_default_graph()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)



