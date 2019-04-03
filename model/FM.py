# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf


class FM():
    def __init__(self, feature_dict, embedding_size=10):
        '''
        初始化和生成FM模型。FM模型只能使用sparse特征，dense特征需要
        先离散化为sparse特征。
        :param feature_dict(dict): python的dict类型，用于保存每种特征的个数
                形式如下：
                { "sparse" : [SingleFeat('uid', 13231), ...],
                  "dense" : [SingleFeat('video_duration', 20), ...]
                }
        The property:
        ==> self.input_indices : tf.placeholder, shape(value_num, 2)
        ==> self.input_indices : tf.placeholder, shape(value_num,)
        ==> self.input_shape   : tf.placeholder, shape(2)
        ==> self.input_Y1      : tf.placeholder, shape(batch_size, )
        ==> self.input_Y2      : tf.placeholder, shape(batch_size, )
        '''
        if "sparse" not in feature_dict:
            print("==> Feature dict does not has key \'sparse\'.")
            exit(1)

        # 得到需要总的embedding vector个数
        self.feat_nums = [i.dimension for i in feature_dict['sparse']]
        self.embed_num = sum(self.feat_nums)
        self.embedding_size = embedding_size
        # Define the placeholder
        self.input_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        self.input_values = tf.placeholder(dtype=tf.float32, shape=[None])
        self.input_shape = tf.placeholder(dtype=tf.int64, shape=[2])
        self.input_Y1 = tf.placeholder(dtype=tf.float32, shape=[None])
        self.input_Y2 = tf.placeholder(dtype=tf.float32, shape=[None])

        # Transform input to SparseTensor
        self.input_sparse = tf.SparseTensor(self.input_indices, self.input_values, self.input_shape)

    def InitFM(self, output_type="finish"):
        '''
        初始化graph图
        :param output: (str)"finish" or "like"
        :return: None
        '''
        # Define weights
        with tf.variable_scope('weights'+output_type):
            W0 = tf.Variable(0.1, name="W0")
            W1 = tf.Variable(tf.truncated_normal([self.embed_num]), name="W1")
            W2 = tf.Variable(tf.truncated_normal([self.embed_num, self.embedding_size]), name="W2")

        # Calculate the linear part
        with tf.variable_scope("linear"+output_type):
            # y_linear shape(batch_size, 1)
            y_linear = tf.sparse_matmul(a=self.input_sparse,
                                        b=W1,
                                        transpose_b=True,
                                        a_is_sparse=True,
                                        name="y_linear")

        # Calculate the cross part
        with tf.variable_scope("cross"+output_type):
            # part1 shape(batch_size, embedding_size)
            part1 = tf.matmul(a=self.input_sparse,
                              b=W2,
                              a_is_sparse=True,
                              name="part1")
            part1_square = tf.square(part1, name="cross1_square")

            # part2 shape(batch_size, embedding_size)
            W2_square = tf.square(W2,name="W2_square")
            input_sparse_square = tf.math.square(self.input_sparse,name="input_square")
            part2 = tf.matmul(a=input_sparse_square,
                              b=W2_square,
                              a_is_sparse=True,
                              name="part2")

            # apply sub
            sub = tf.subtract(part1_square, part2, name="part1_part2_sub")

            # get cross output, shape(batch_size, 1)
            cross_out = 0.5 * tf.reduce_sum(sub, reduction_indices=1,
                                            keep_dims=True,
                                            name="cross_out")

        # Calculate the final result
        with tf.variable_scope("output"+output_type):
            logits = W0 + y_linear + cross_out

        return logits

    def train(self, train_data_input, batch_size=1024, epochs=1, validation_frac=0.1):
        '''
        给定模型参数训练模型
        :param train_data_input:
        :return: None
        '''









