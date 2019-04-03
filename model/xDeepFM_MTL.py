import tensorflow as tf
from deepctr.input_embedding import preprocess_input_embedding
from deepctr.layers.core import MLP, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_fun
from deepctr.utils import check_feature_config_dict

'''
模型函数输入：
==> feature_dim_dict : 特征输入维度字典
==> embedding_size : embedding的维度
==> hidden_size : 隐藏层size
==> cin_layer_size : CIN层的size
==> cin_split_half :
==> task_net_size : 
==> l2_reg_linear : L2正则化系数
==> l2_reg_embedding : L2正则化系数
==> seed : 生成随机数时的种子
'''
def xDeepFM_MTL(feature_dim_dict, embedding_size=8, hidden_size=(256, 256), cin_layer_size=(256, 256,),
                cin_split_half=True,
                task_net_size=(128,), l2_reg_linear=0.000001, l2_reg_embedding=0.000001,
                seed=1024, ):
    check_feature_config_dict(feature_dim_dict)        # 未知
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    deep_emb_list, linear_logit, inputs_list = preprocess_input_embedding(
        feature_dim_dict, embedding_size, l2_reg_embedding, l2_reg_linear, 0.0001, seed)

    # video_input = tf.keras.layers.Input((128,))
    # inputs_list.append(video_input)
    fm_input = concat_fun(deep_emb_list, axis=1)   # 模型输入
    '''
    构建CIN，默认CIN的size为[256,256]，激活函数为relu，输入为
    (batch_size,field_size,embedding_size)，输出为(batch_size,feature_num)。
    如果split_half为True，那么隐藏层的feature map只有一半的会连接到输出单元。
    '''
    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)  # 全连接输出到Output_unit

    '''
     Flatten将输入除了batch的维度，其他维度拉直，得到的输出为(batch_size, sum_size)
     将embedding特征直接输入MLP
     '''
    deep_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = MLP(hidden_size)(deep_input)
    
    '''
     将deep_out过一个MLP，并全连接到finish的logits输出，同样的操作应用于like的logits输出
     '''
    finish_out = MLP(task_net_size)(deep_out)
    finish_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out)

    like_out = MLP(task_net_size)(deep_out)
    like_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out)
    '''
     最终的finish的logit由linear_logit,finish_logit\like_logit和exFM_logit三者叠加。
     '''
    finish_logit = tf.keras.layers.add(
        [linear_logit, finish_logit, exFM_logit])
    like_logit = tf.keras.layers.add(
        [linear_logit, like_logit, exFM_logit])
    '''
     将logit通过sigmoid转化为概率，通过输入和输出构建model
     '''
    output_finish = PredictionLayer('sigmoid', name='finish')(finish_logit)
    output_like = PredictionLayer('sigmoid', name='like')(like_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[
                                  output_finish, output_like])
    return model
