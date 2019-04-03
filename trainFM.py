# -*- coding : utf-8 -*-
import os
import sys
from logger import Logger
import xlearn as xl
from config import Config

conf = Config()
sys.stdout = Logger(os.path.join(conf.log_path, conf.model+"_log.log"), sys.stdout)  # 将控制台输出同时输出到log文件

if __name__ == "__main__":
    # 设定sparse features和dense features
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', 'gender']
    dense_features = ['video_duration', 'beauty']
    target = ['finish', 'like']
    epochs = 5
    split_num = 1

    for tar in target:
        #if tar == "like": continue
        print("==> Start to train, target : %s" % tar)
        valid_path = os.path.join(conf.data_path, "FFM_" + tar + "_valid.txt")
        model_path = os.path.join(conf.model_save_path, "FFM_"+tar+".model")

        print("==> Training model")
        for epoch in range(epochs):
            print("==> Epochs %d in all, training epoch : %d" % (epochs, epoch+1))
            # 设置训练属性和参数
            param = {'task': 'binary',
                     'k': 5,
                     'lr': 0.2 ,
                     'lambda': 0.003,
                     'metric': 'auc',
                     'epoch': 100,
                     'nthread': 6,
                     'block_size': 10000}
            for i in range(split_num):
                train_path = os.path.join(conf.data_path, "FFM_" + tar + "_train%d.txt"%(i+1))
                ffm_model = xl.create_ffm()
                ffm_model.setTrain(train_path)
                ffm_model.setValidate(valid_path)
                ffm_model.setOnDisk()
                if os.path.exists(model_path):
                    ffm_model.setPreModel(model_path)
                ffm_model.fit(param, model_path)
    print("==> Finish training")
