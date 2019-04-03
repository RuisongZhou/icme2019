# -*- coding : utf-8 -*-
import os
import time
import pickle
import pandas as pd
from dataloader import Dataloader
from model import xDeepFM_MTL
from config import Config

conf = Config()

if __name__ == "__main__":
    # 模型使用的sparse feature列表和dense feature列表，以及学习目标的列表
    print("==> Get sparse features and dense features")
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did']  # 稀疏的特征，category类型特征
    dense_features = ['video_duration']  # dense的特征，embedding式特征
    target = ['finish', 'like']

    # 读取数据以及处理数据
    loader = Dataloader(train_path=os.path.join(conf.data_path, "train.csv"),
                        test_path=os.path.join(conf.data_path, "test.csv"),
                        valid_path=os.path.join(conf.data_path, "valid.csv"))
    loader.LoadData()

    '''
    训练集输入和测试集输入，其中DataFrame.values返回numpy.array类型的值，如果返回
    一列值，得到的是一维向量，如果是多列的值，那么是二维矩阵。下面应该返回的都是
    一维向量
    '''
    print("==> Loading feature list")
    sparse_feature_list = pickle.load(open(os.path.join(conf.data_path, "sparse_features.pkl"), "rb"))
    dense_feature_list = pickle.load(open(os.path.join(conf.data_path, "dense_features.pkl"), "rb"))
    print("==> Generate training input and test input")
    train_model_input = [loader.train_data[feat.name].values for feat in sparse_feature_list] + \
                        [loader.train_data[feat.name].values for feat in dense_feature_list]
    train_labels = [loader.train_data[target[0]].values, loader.train_data[target[1]].values]
    if not conf.train:  # 如果为测试模式，那么需要得到测试集输入
        test_model_input = [loader.test_data[feat.name].values for feat in sparse_feature_list] + \
                            [loader.test_data[feat.name].values for feat in dense_feature_list]
        test_labels = [loader.test_data[target[0]].values, loader.test_data[target[1]].values]

    '''
    构建模型，传递含有sparse特征名和dense特征名的字典，返回tf.keras.Model类。
    然后compile模型训练参数
    '''
    print("==> Build model and compile")
    model = xDeepFM_MTL.xDeepFM_MTL({'sparse': sparse_feature_list,
                             'dense': dense_feature_list})
    model.compile("adagrad", "binary_crossentropy", loss_weights=conf.loss_weights)

    '''
    在训练集上迭代训练模型指定的轮数。如果是训练模式，那么验证集为手动分割的数据，
    如果为测试模式，那么测试集为测试输入。
    x : 输入
    y : 输出
    batch_size :
    epochs : 训练轮数
    verbose : 0,1,2，进度显示形式
    '''
    if conf.train:
        print("==> Training model")
        history = model.fit(train_model_input, train_labels, batch_size=conf.batch_size,
                            epochs=conf.epochs, verbose=1, validation_split=conf.validation_frac)
    else:
        print("==> Training model")
        history = model.fit(train_model_input, train_labels, batch_size=conf.batch_size,
                            epochs=conf.epochs, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2**14)
    print("==> Saving the model")
    timearray = time.localtime(time.time())   # 给模型文件加上时间戳
    timestr = time.strftime("%Y%m%d%H%M%S", timearray)
    model.save(os.path.join(conf.model_save_path, "model_" + timestr + ".model"))
    
    '''
    将测试集保存为指定的形式用于提交
    '''
    if not conf.train:
        temp_test = pd.read_csv(conf.test_data_path, sep='\t', names=[
            'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
            'like', 'music_id', 'did', 'creat_time', 'video_duration'])
        result = temp_test[["uid", 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish':'finish_probability',
                               'like':'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result.to_csv(os.path.join(conf.result_save_path, "result_" + timestr + ".csv"),
                      index=None, float_format='%.6f')

