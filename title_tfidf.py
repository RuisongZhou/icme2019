# -*- coding : utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from config import Config

conf = Config()
def title_preprocess(filename):
    '''
    将title特征提取使用tfidf转化
    :param filename:
    :return:
    '''
    item_id = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            lines = lines.strip('\n')
            if(lines):
                lines = json.loads(lines)
                item_id.append(lines['item_id'])
                face_attrs = lines['title_features']
                people_num = len(face_attrs)
                if people_num == 0:
                    gender.append(None)
                    beauty.append(None)
                    relative_position.append(None)
                else:
                    gender_num = 0;
                    beauty_num = 0;
                    relative_position_num = 0;
                    for each in face_attrs:
                        gender_num += each['gender']
                        beauty_num += each['beauty']
                        relative_position_num += sum(each['relative_position'])
                    beauty_num /= people_num
                    relative_position_num /= people_num
                    gender_num /= people_num
                    gender.append(gender_num)
                    beauty.append(beauty_num)
                    relative_position.append(relative_position_num)
            if not lines:
                break