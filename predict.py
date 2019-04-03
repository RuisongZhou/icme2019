# -*- coding: utf-8 -*-

import os
import xlearn as xl
from config import Config
from utils import GenResultFromTxt

conf = Config()

if __name__=="__main__":
    # sparse feature and dense features
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', 'gender']
    dense_features = ['video_duration', 'beauty']
    target = ['finish', 'like']

    # 将训练集转化为libsvm形式
    for tar in target:
        test_path = os.path.join(conf.data_path, "FFM_" + tar + "_test.txt")
        model_path = os.path.join(conf.model_save_path, "FFM_"+tar+".model")
        save_path = os.path.join(conf.result_save_path, "FFM_"+tar+".output")
        print("==> Predict model")
        fm_model = xl.create_fm()
        fm_model.setTest(test_path)
        fm_model.setSigmoid()
        fm_model.predict(model_path, save_path)

    print("==> Generate result file")
    finish_output_path = os.path.join(conf.result_save_path, "FFM_finish.output")
    like_output_path = os.path.join(conf.result_save_path, "FFM_like.output")
    result_save_path = os.path.join(conf.result_save_path, "FFM_result322.csv")
    GenResultFromTxt(conf.test_data_path, finish_output_path, like_output_path, result_save_path)
    print("==> Finish generating result file")