# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from config import Config
from utils import CheckPath, DataProcess

conf = Config()

def GenerateData(final_track_path, test_data_path, sparse_features, dense_features, save_path,
                 track_face_path=None, track_title_path=None, track_video_path=None,
                 valid_frac=None, transformer_path=None, regen=False, feature_path=None):
    '''
    用于生成训练所需要的数据，保存为CSV格式的文件
    :param final_track_path:    (str)final_track_path文件路径
    :param test_data_path:      (str)测试集文件路径
    :param sparse_features:     (list)sparse feature列表
    :param dense_features:      (list)dense feature列表
    :param save_path:           (str)CSV文件保存路径
    :param track_face_path:     (str)人脸特征文件路径
    :param track_title_path:    (str)视频标题特征文件路径
    :param track_video_path:    (str)视频视觉特征文件路径
    :param valid_frac:          (float)0~1之间的值，验证集比例
    :param transformer_path:    (str)转化文件路径
    :param regen:               (bool)是否重新生成转换器
    :param feature_path:        (str)(feat_name, feat_dim)二进制文件保存路径
    :return:
    '''
    processor = DataProcess()
    print("==> Loading training data %s" % os.path.split(final_track_path)[-1])
    data = pd.read_csv(final_track_path, sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
        'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    train_size = data.shape[0]

    print("==> Loading testing data %s" % os.path.split(test_data_path)[-1])
    test_data = pd.read_csv(test_data_path, sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
        'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    data = data.append(test_data)

    # 开始对数据进行处理
    print("==> Begin to process data")
    processor.MergeFeatures(data, track_face_path, track_title_path, track_video_path)
    processor.Fillna(data, sparse_features, dense_features)
    processor.EncodeAndRegular(data, sparse_features, dense_features, save_path=transformer_path, regen=regen)
    processor.GenerateFeatureList(data, sparse_features, dense_features, save_path=feature_path)
    # 保存文件
    print("==> Begin to save data")
    train_path = os.path.join(save_path, "train.csv")
    test_path = os.path.join(save_path, "test.csv")
    data.iloc[train_size:].to_csv(test_path, index=None, encoding="utf-8")
    if valid_frac is not None:
        print("==> Using valid data")
        valid_path = os.path.join(save_path, "valid.csv")
        split_index = int(train_size * (1 - valid_frac))
        data.iloc[:split_index].to_csv(valid_path, index=None, encoding="utf-8")
        data.iloc[split_index:train_size].to_csv(train_path, index=None, encoding="utf-8")
    else:
        data.iloc[:train_size].to_csv(train_path, index=None, encoding="utf-8")
    print("==> Successfully saving file")

class Dataloader():
    def __init__(self, train_path, test_path, valid_path=None):
        '''
        用于读取数据用于模型的输入
        :param train_path: 训练集的csv格式输入文件路径
        :param test_path: 测试集csv格式输入文件路径
        :param valid_path: 验证集csv格式输入路径
        '''
        CheckPath(train_path)
        CheckPath(test_path)
        if valid_path is not None:
            CheckPath(valid_path)

        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path

    def LoadData(self):
        self.train_data = pd.read_csv(self.train_path, encoding="utf-8")
        self.test_data = pd.read_csv(self.test_path, encoding="utf-8")
        if self.valid_path is not None:
            self.valid_data = pd.read_csv(self.valid_path, encoding="utf-8")


if __name__ =='__main__':
    conf = Config()

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', 'gender']
    dense_features = ['video_duration', 'beauty']
    target = ['finish', 'like']
    GenerateData(conf.final_track2_train_path,conf.test_data_path, sparse_features, dense_features,
                 conf.process_path,track_face_path=conf.track2_face_attrs_path, track_title_path=conf.track2_title_path)
