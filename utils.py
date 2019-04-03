# -*- coding : utf-8 -*-
'''
utils.py文件用于对数据进行处理和转换，主要函数和功能如下：
==> CheckPath()           : 检查文件路径是否存在
==> GenResultFormTxt()    : 根据生成的finish.output和like.output文件与测试集合并成result文件
==> DataProcess()         : 用于对数据进行处理的类，包括特征合并、填充空值、one-hot编码和归一化、产生FeatureList
==> GenProcessData()      : 用于产生FM模型使用的libsvm格式文件
'''
import os
import pickle
import pandas as pd
from config import Config
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 构建namedtuple
Feature = namedtuple(typename="Feature" , field_names=['name', 'dimension'])

def CheckPath(path):
    if not os.path.exists(path):
        print("==> File does not exist : %s" % os.path.split(path)[-1])
        exit(1)

def GenResultFromTxt(test_data_path, finish_txt_path, like_txt_path, result_save_path):
    '''
    根据xLearn生成的结果Txt文件与测试集的数据进行Merge得到最后的提交文件，
    结果txt文件的格式为：
    -------------------txt-----------------------
    0.1234
    0.2344
    0.5342
    ...
    ---------------------------------------------
    :param test_data_path: 测试集文件
    :param finish_txt_path: xLearn生成的finish列的txt文件
    :param like_txt_path: xLearn生成的like列的txt文件
    :param result_save_path: 结果保存路径
    :return: None
    '''
    CheckPath(test_data_path)
    CheckPath(finish_txt_path)
    CheckPath(like_txt_path)

    # 读取测试集
    print("==> Loading testing data")
    test_data = pd.read_csv(test_data_path, sep="\t", names=[
                'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
                'like', 'music_id', 'did', 'creat_time', 'video_duration'
            ])
    result = test_data[["uid", "item_id", "finish", "like"]].copy()
    result.rename(columns={'finish':'finish_probability',
                               'like':'like_probability'}, inplace=True)

    # 读取finish和like的txt文件
    print("==> Loading finish and like txt file")
    with open(finish_txt_path, "r") as fp:
        finish_list = fp.read().split()
    with open(like_txt_path, "r") as fp:
        like_list = fp.read().split()

    # 组合结果
    print("==> Merge the result")
    result["finish_probability"] = finish_list
    result["like_probability"] = like_list
    result.to_csv(result_save_path, index=None, float_format='%.6f')

class DataProcess():
    def __init__(self):
        print("==> Init DataProcess class.")

    def MergeFeatures(self, data, track_face_path=None, track_title_path=None,
                      track_video_path=None, face_dict_path=None):
        '''
        用于将可用的features进行合并成一个csv文件，其中字段定义如下：
        .csv文件：
        ==> uid,user_city...,video_duration : 同final_track文件
        ==> beauty(float) : 来自于face feature
        ==> gender(int) : 来自于face feature
        ==> relative_position1-4(float) : 来自于face feature，是相对长宽距离
        ==> title_words(list) : 来自于title feature，词的index列表
        ==> title_freqs(list) : 来自于title feature，词的个数列表，与index列表对应的词相互对应
        :param data:              (DataFrame)csv格式文件，必须传递原始的文件
        :param track_face_path:   (str)人脸特征路径
        :param track_title_path:  (str)视频title特征
        :param track_video_path:  (str)视频的embedding特征
        :return: None
        '''
        if track_face_path is not None:
            '''
            合并face特征到特征csv文件中，步骤为：
            ==> 读取track_face_path文件
            ==> 转化成行list
            ==> 将每一行使用eval()函数转化成字典
            ==> 整合成一个字典
            ==> 使用key进行检索并加入特征
            '''
            if face_dict_path is None or not os.path.exists(face_dict_path):
                print("==> Loading training data %s" % os.path.split(track_face_path)[-1])
                with open(track_face_path, "r") as fp:
                    content = fp.read().split("\n")
                print("==> Trans to list")
                content = [eval(i) for i in content if i != ""]  # 转化成dict的list
                print("==> Trans to tuple")
                content = [(i["item_id"], i["face_attrs"]) for i in content] # 转化成(item_id,attrs)的tuple形式
                content = dict(content)   # 转化成dict形式
                for key in content:
                    for num in range(len(content[key])):
                        if num == 0 : pass
                        else:
                            content[key][0]["beauty"] += content[key][num]["beauty"]
                            content[key][0]["gender"] += content[key][num]["gender"]
                            content[key][0]["relative_position"][0] += content[key][num]["relative_position"][0]
                            content[key][0]["relative_position"][1] += content[key][num]["relative_position"][1]
                            content[key][0]["relative_position"][2] += content[key][num]["relative_position"][2]
                            content[key][0]["relative_position"][3] += content[key][num]["relative_position"][3]
                    # content[key][0]["gender"] /= len(content[key])

                if os.path.exists(face_dict_path):
                    print("==> Saving face feature dict")
                    pickle.dump(content, open(face_dict_path, "wb"))
            else:
                print("==> Loading face feature dict")
                content = pickle.load(open(face_dict_path, "rb"))
            data["beauty"] = data["item_id"].map(lambda x: None if (x not in content) or (len(content[x])==0)
                                                                                else content[x][0]["beauty"])
            data["gender"] = data["item_id"].map(lambda x: None if (x not in content) or (len(content[x])==0)
                                                                                else content[x][0]["gender"])
            data["relative_position"] = data["item_id"].map(lambda x: None if (x not in content) or len(content[x])==0
                                                                                else content[x][0]["relative_position"][0] +
                                                                                     content[x][0]["relative_position"][1] +
                                                                                     content[x][0]["relative_position"][2] +
                                                                                     content[x][0]["relative_position"][3])

            del content  # 删除content以节省空间
        if track_title_path is not None:
            '''
            合并title特征到csv文件中，步骤为: 同上
            '''
            print("==> Loading training data %s" % os.path.split(track_title_path)[-1])
            with open(track_title_path, "r") as fp:
                content = fp.read().split("\n")
            content = [eval(i) for i in content if i != ""]
            content = [(i["item_id"], i["title_features"]) for i in content]
            content = dict(content)
            data["title_words"] = data["item_id"].map(lambda x: None if x not in content
                                                                            else list(content[x].keys()))
            data["title_freqs"] = data["item_id"].map(lambda x: None if x not in content
                                                                            else list(content[x].values()))
            del content  # 删除content以节省空间
        if track_video_path is not None:
            pass # TODO

        print("==> Finish merging features")

    def AddFeatures(self, data):
        add_sparse_feat = []
        if "uid" in data.columns and "author_id" in data.columns:
            print("==> Add feature >>> same_id <<<")
            data["same_id"] = list(map(lambda x,y: 1 if (x is not None) and (y is not None) and (x==y) else 0, data["uid"], data["author_id"]))
            add_sparse_feat.append("same_id")
        if "user_city" in data.columns and "item_city" in data.columns:
            print("==> Add feature >> same_city <<<")
            data["same_city"] = list(map(lambda x,y: 1 if (x is not None) and (y is not None) and (x==y) else 0, data["user_city"], data["item_city"]))
            add_sparse_feat.append("same_city")

    def Fillna(self, data, sparse_features, dense_features):
        '''
        对csv中的空特征进行填充，sparse特征填充-1，dense特征填充0
        :param data: csv文件
        :param sparse_features: sparse特征名列表
        :param dense_features: dense特征名列表
        :return: None
        '''
        print("==> Fillna for sparse and dense features")
        data[sparse_features] = data[sparse_features].fillna(-1,)
        data[dense_features] = data[dense_features].fillna(data[dense_features].mean())

    def EncodeAndRegular(self, data, sparse_features, dense_features, save_path=None, regen=False):
        '''
        对sparse features进行LabelEncode，并保存LabelEncoder二进制文件
        之后使用。对dense features进行归一化，并保存转化的二进制文件用于
        之后使用
        :param data: csv文件
        :param sparse_features: 稀疏特征特征名列表
        :param dense_features: dense特征特征名列表
        :param save_path: 保存二进制文件路径
        :param regen: 是否重新构造transformer文件
        :return: None
        '''
        print("==> Apply label encoder and regularization")
        # 对sparse feature进行处理
        for feat in sparse_features:
            if save_path:     # 如果存在保存路径
                lbe_path = os.path.join(save_path, "lbe_"+feat+"_train.pkl")
                if os.path.exists(lbe_path) and not regen:  # 如果保存路径存在pkl文件，pickle.load
                    print("==> Loading %s label encoder" % feat)
                    lbe = pickle.load(open(lbe_path, "rb"))
                    data[feat] = lbe.transform(data[feat])
                else:     # 如果保存路径不存在pkl文件
                    lbe = LabelEncoder()
                    data[feat] = lbe.fit_transform(data[feat])
                    pickle.dump(lbe, open(lbe_path, "wb"))
                    print("==> Saving %s label encoder." % feat)
            else:   # 如果没有保存路径，一次性生成和使用
                lbe = LabelEncoder()
                data[feat] = lbe.fit_transform(data[feat])
        # 对dense features进行处理
        for feat in dense_features:
            if feat in ["beauty", "relative_position"]: continue
            if save_path:
                mms_path = os.path.join(save_path, "mms_" + feat+"_train.pkl")
                if os.path.exists(mms_path) and not regen:
                    print("==> Loading %s Minmax scaler." % feat)
                    mms = pickle.load(open(mms_path, "rb"))
                    data[[feat]] = mms.transform(data[[feat]])
                else:
                    mms = MinMaxScaler(feature_range=(0, 1))
                    data[[feat]] = mms.fit_transform(data[[feat]])
                    pickle.dump(mms, open(mms_path, "wb"))
                    print("==> Saving %s Minmax scaler" % feat)
            else:
                mms = MinMaxScaler(feature_range=(0, 1))
                data[[feat]] = mms.fit_transform(data[[feat]])

    def GenerateFeatureList(self, data, sparse_features, dense_features, save_path=None):
        '''
        该函数用于产生sparse_feature_list和dense_feature_list，列表元素为
        Feature(name, dimension)类型。
        Feature 是一个带有签名的namedtuple，Feature(name, dimension)
        ==> name : 特征名称
        ==> dimension : 独特特征的个数(对于稀疏特征)，任何值(对于dense特征)
        :param data: csv的数据
        :param sparse_features: sparse feature的特征名列表
        :param dense_features: dense feature的特征名列表
        :param save_path: feature list的保存路径
        :return: sparse_feature_list , dense_feature_list
        '''
        print("==> Generate feature list")
        sparse_feature_list = [Feature(feat, data[feat].nunique()) for feat in sparse_features]
        dense_feature_list = [Feature(feat, 0) for feat in dense_features]
        if save_path:
            try:
                pickle.dump(sparse_feature_list, open(os.path.join(save_path,"sparse_features.pkl"), "wb"))
                pickle.dump(dense_feature_list, open(os.path.join(save_path,"dense_features.pkl"), "wb"))
                print("==> Successfully saving feature list")
            except:
                print("==> Failed saving feature list")
        return sparse_feature_list, dense_feature_list


    def check_data(self,data):
        '''
        清洗训练数据
        :param data:
        :return:
        '''
        print("==> checking train data")
        data = data[(data['video_duration'] <= 80) & (data['video_duration'] >= 5)]
        data = data[(data['finish'] == 1) | (data['finish'] == 0)]
        data = data[(data['like'] == 1) | (data['like'] == 0)]
        print("==> data shape is ")
        print(data.shape)
        return data
def GenProcessData(conf, final_track_path, test_data_path, sparse_features, dense_features, target,
                   track_face_path=None, track_title_path=None, track_video_path=None):
    '''
    用于产生FM模型使用的libSVM格式训练集，保存在conf.process_path文件夹内
    :param conf: 参数配置文件
    :param sparse_features: sparse features列表
    :param dense_features: dense features列表
    :param target: target列表
    :return: None
    '''
    print("==> Init DataProcess")
    processor = DataProcess()

    print("==> Loading training data")
    data = pd.read_csv(final_track_path, sep='\t', names=[
            'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
            'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    data = processor.check_data(data)
    train_size = len(data)
    print("==> Loading testing data")
    test_data = pd.read_csv(test_data_path, sep='\t', names=[
            'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
            'like', 'music_id', 'did', 'creat_time', 'video_duration'])
    data = data.append(test_data)

    # 处理data
    print("==> Begin to process data")
    processor.MergeFeatures(data, track_face_path, track_title_path, track_video_path,
                            face_dict_path=os.path.join(conf.process_path, "face_dict.pkl"))
    processor.Fillna(data, sparse_features, dense_features)
    processor.EncodeAndRegular(data, sparse_features, dense_features, save_path=conf.transformer_path, regen=False)
    processor.GenerateFeatureList(data, sparse_features, dense_features, save_path=conf.process_path)

    print("==> Saving train data and test data")  # 只取出需要使用的特征
    data.iloc[:train_size][sparse_features+dense_features+target].to_csv(os.path.join(conf.process_path, "train.csv"), index=None, encoding="utf-8")
    data.iloc[train_size:][sparse_features+dense_features+target].to_csv(os.path.join(conf.process_path, "test.csv"), index=None, encoding="utf-8")

# 处理数据集成输入模型格式，LR和FM模型支持CSV格式和libsvm格式
# FFM的输入格式只支持libffm的格式
# libsvm format :
#    label index_1:value_1 index_2:value_2 ... index_n:value_n
# CSV format :
#    value_1 value_2 ... value_n label
# libffm format :
#    label field_1:index_1:value_1 field_2:index_2:value_2 ...
# 上述格式中的分隔符也可以是","
class FileFormatTransformer():
    def __init__(self, train_path, test_path, sparse_feature_list, dense_feature_list,
                 target, validation_frac, split_num, save_path, model):
        '''
        :param train_path:           处理后的训练集csv文件路径
        :param test_path:            处理后的测试集csv文件路径
        :param sparse_feature_list:  csv文件的稀疏特征列表
        :param dense_feature_list:   csv文件的dense特征列表
        :param target:               训练的目标
        :param validation_frac:      验证集的比例
        :param split_num:            分割训练集个数
        :param save_path:            处理文件的保存路径
        :param model:                模型名称(FM or FFM)
        '''
        # Check path
        CheckPath(train_path)
        CheckPath(test_path)

        self.train_path = train_path
        self.test_path = test_path
        self.sparse_feature_list = sparse_feature_list
        self.dense_feature_list = dense_feature_list
        self.target = target
        self.validation_frac = validation_frac
        self.split_num = split_num
        self.save_path = save_path
        self.model = model

    def GenerateLibSVMData(self):
        '''
        将CSV文件转化为LibSVM文件格式
        :return: None
        '''
        # 将测试集转化为libsvm文件格式
        print("==> Transform test data to libsvm")
        test_data = pd.read_csv(self.test_path, encoding="utf-8")
        test_path = [os.path.join(self.save_path, "%s_%s_test.txt"%(self.model, tar)) for tar in self.target]
        TransformTxt(test_data, self.sparse_feature_list, self.dense_feature_list, self.target, test_path, format=self.model)
        # 将训练集转化为libsvm文件格式
        if type(self.train_path) == str:  # 如果训练集为一个csv文件
            print("==> Loading training data")
            data = pd.read_csv(self.train_path, encoding="utf-8")
            print("==> Shuffle training data")
            data = data.sample(frac=1.0).reset_index(drop=True)
            train_size = int(len(data) * (1 - self.validation_frac))
            train_data = data.iloc[:train_size]
            valid_data = data.iloc[train_size:]
            del data
            # 转化验证集
            print("==> Transform valid data to libsvm")
            valid_path = [os.path.join(self.save_path, "%s_%s_valid.txt"%(self.model, tar)) for tar in self.target]
            TransformTxt(valid_data, self.sparse_feature_list, self.dense_feature_list, self.target, valid_path, format=self.model)
            # 转化训练集
            print("==> Transform train data to libsvm")
            batch_num = int(len(train_data) / self.split_num)
            for i in range(self.split_num):
                start = i * batch_num
                end = (i + 1) * batch_num
                train_path = [os.path.join(self.save_path, "%s_%s_train%d.txt"%(self.model, tar, i+1)) for tar in self.target]
                TransformTxt(train_data.iloc[start:end], self.sparse_feature_list, self.dense_feature_list, self.target, train_path, format=self.model)
            print("==> Finish transformation")
        elif type(self.train_path) == list:  # 如果训练集为一个列表，多个csv文件
            print("==> Do not support list.")  # TODO
        else:
            print("==> Train path type error.")
            exit(1)

    def merge_files(self):
        '''
        将生成的多个训练集文件合并成一个训练集，并储存在当前文件中
        :return: none
        '''
        for tar in self.target:
            all_data = pd.DataFrame()
            file_path = [os.path.join(self.train_path, "%s_%s_train%d.txt"%(self.model, tar, i+1 )) for i in range(self.split_num)]
            for file in file_path:
                data = pd.read_csv(file, sep=',', header=None)
                all_data = all_data.append(data)
            save_path = os.path.join(self.save_path, "%s_%s_train_all.txt"%(self.model, tar))
            all_data.to_csv(save_path, sep=',', header=None, index=None, encoding="utf-8")

def TransformTxt(csv_data, sparse_feature_list, dense_feature_list, target, save_path, format="FM"):
    '''
    用于将CSV文件转化为其他格式形式，用于模型输入，libSVM文件形式为
      label index_1:value_1 index_2:value_2 ...
    其中label为标签，index为将样本特征展开之后的index，value如果
    特征为one-hot特征，那么值为1，否则，值为dense的值。
    csv文件的每一sparse feature列都已经通过LabelEncoder的到index值
    :param csv_data: CSV文件
    :param sparse_feature_list: 稀疏特征列表
    :param dense_feature_list: dense特征列表
    :param target: 学习目标('finish' 或 'like')
    :param save_path: 转化后的文件保存路径
    :param format: 转化的文件格式
    :return: None
    '''
    sparse_features = [i.name for i in sparse_feature_list]
    sparse_feature_num = [i.dimension for i in sparse_feature_list]
    sparse_feature_sum = sum(sparse_feature_num)
    dense_features = [i.name for i in dense_feature_list]

    # csv_data[sparse_features].replace({-1: None}, inplace=True)   # 将-1替换回None
    # csv_data[dense_features].replace({0, None}, inplace=True)   # 将0替换回None

    # 对索引值进行求和
    sparse_feature_num = [0] + sparse_feature_num[:-1]
    for i in range(1, len(sparse_feature_num)):
        sparse_feature_num[i] = sparse_feature_num[i-1] + sparse_feature_num[i]
    print("==> Applying transformation for sparse features")
    # 对sparse feature进行处理
    if format == "FM":
        print("\t", sparse_feature_num)
        print("==> Transform to libsvm format")
        for feat, num in zip(sparse_features, sparse_feature_num):
            csv_data[feat] = csv_data[feat].apply(lambda x: None if x is None else "{}:1".format(int(x)+num))
    elif format == "FFM":
        print("==> Transform to libffm format")
        for feat, field, num in zip(sparse_features, range(len(sparse_features)), sparse_feature_num):
            print("==> \tTransform feature %s" % feat)
            csv_data[feat] = csv_data[feat].apply(lambda x: None if x is None else "{}:{}:1".format(field, int(x)))
    else:
        print("==> Error : %s format is not supported."%format)
        exit(1)
    # 对dense features进行处理
    print("==> Applying transformation for dense features")
    if format == "FM":
        for feat in dense_features:
            csv_data[feat] = csv_data[feat].apply(lambda x: None if x is None else "{}:{}".format(sparse_feature_sum, float(x)))
            sparse_feature_sum += 1
    elif format == "FFM":    # 对于dense features，每个feature只有一个特征
        for feat in dense_features:
            print("==> Transform feature %s" % feat)
            field += 1
            csv_data[feat] = csv_data[feat].apply(lambda x: None if x is None else "{}:{}:{}".format(field, sparse_feature_sum, float(x)))
            sparse_feature_sum += 1
    else:
        print("==> Error : %s format is not supported." % format)
        exit(1)

    # 保存文件
    csv_data[[target[0]]+sparse_features+dense_features].to_csv(save_path[0], sep=',', header=None, index=None, encoding="utf-8")
    csv_data[[target[1]]+sparse_features+dense_features].to_csv(save_path[1], sep=',', header=None, index=None, encoding="utf-8")


if __name__ == "__main__":
    conf = Config()

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', 'gender']
    dense_features = ['video_duration', 'beauty', 'relative_position']
    target = ['finish', 'like']
    GenProcessData(conf, conf.final_track2_train_path, conf.test_data_path, sparse_features, dense_features,
                  ["finish", "like"], track_face_path=conf.track2_face_attrs_path)
    sparse_feature_list = pickle.load(open(os.path.join(conf.process_path, "sparse_features.pkl"), "rb"))
    dense_feature_list = pickle.load(open(os.path.join(conf.process_path, "dense_features.pkl"), "rb"))
    transformer = FileFormatTransformer(conf.process_path+'train.csv', conf.process_path+'test.csv', sparse_feature_list,
                                        dense_feature_list, target, 0.8, 1, conf.data_path, 'FFM')
    transformer.GenerateLibSVMData()
    #transformer.merge_files()