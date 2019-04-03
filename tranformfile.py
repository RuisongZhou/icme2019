# -*- coding : utf-8 -*-

'''
该文件用于将原始训练集文件导出为各种格式的文件，用于
训练调用
'''
import os
import pickle
import pandas as pd
from config import Config
from dataloader import *
from utils import TransformToLibSVM

conf = Config()

if __name__ == "__main__":
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', 'gender']
    dense_features = ['video_duration', 'beauty']
    target = ['finish', 'like']

    # 产生FM的训练文件
    GenProcessData(conf, sparse_features, dense_features, target)

    sparse_feature_list = pickle.load(open(os.path.join(conf.process_path, "sparse_list.pkl"), "rb"))
    dense_feature_list = pickle.load(open(os.path.join(conf.process_path, "dense_list.pkl"), "rb"))
    print("==> Finish")

    MergeTxt(conf, target, split_num=5)




