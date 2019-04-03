# -*- coding : utf-8 -*-

import os

class Config():
    def __init__(self):
        # 训练集文件路径
        self.test_data_path = "/home/zrs/data/ICME2019/test/final_track2_test_no_anwser.txt"
        self.final_track2_train_path = "/home/zrs/data/ICME2019/train/final_track2_train.txt"
        self.track2_face_attrs_path = "/home/zrs/data/ICME2019/train/track2_face_attrs.txt"
        self.track2_title_path = "/home/zrs/data/ICME2019/train/track2_title.txt"
        self.track2_video_features_path = "/home/zrs/data/ICME2019/train/track2_video_features.txt"
        # 其他中间文件保存路径
        # ==> process_path : 处理的训练的文件保存路径，该文件用于模型输入
        # ==> transformer_path : 对sparse feature和dense feature进行处理的类二进制文件
        # ==> model_save_path : 模型保存文件路径
        # ==> result_save_path : 结果文件保存路径
        # ==> data_path : 数据路径(临时、较小的数据文件)
        # ==> log_path : log文件保存路径
        self.process_path = "/home/zrs/data/ICME2019/process/"
        self.transformer_path = "./source/transformer/"
        self.model_save_path = "./source/models/"
        self.result_save_path = "./result/"
        self.data_path = "./data/"
        self.log_path = "./source/log/"

        # 训练参数
        self.model = "FM"  # FM or FFM or Linear
        self.loss_weights = [0.7, 0.3]  # loss的权重分配
        self.validation_frac = 0.1  # 验证集比例
        self.train = False           # 训练验证或训练测试
        self.retrain = False        # 重新训练表示重新生成中间文件
        self.batch_size = 1024      # 训练batch_size
        self.epochs = 1             # 训练epochs数

        self.CheckPath()

    def CheckPath(self):
        '''
        检查路径是否存在，如果不存在，那么新建
        :return: None
        '''
        if not os.path.exists(self.transformer_path):
            os.makedirs(self.transformer_path)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.result_save_path):
            os.makedirs(self.result_save_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)