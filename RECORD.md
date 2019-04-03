# 模型训练预测记录表

**说明**：

```TXT
model                       : 模型文件
batch_size                  : 训练输入batch_size
epochs                  	: 训练轮数
optimizer               	: 训练优化器
loss                    	: 使用的loss
loss_weights            	: finish和like的loss比例
sparse features         	: 使用的sparse feature列名
sparse feature process  	: 对sparse features使用的预处理方法
dense features          	: 使用的dense feature列名
dense feature process   	: 对dense features使用的预处理方法
embedding feature       	: 使用的嵌入式向量特征
embedding feature process   : 使用的嵌入式特征预处理方法
target 						: 训练的目标/标签
finish loss 				: 训练后finish的loss
like loss 					: 训练后like的loss
save model file 			: 保存的训练模型文件
save result file 			: 保存的测试结果文件
submit finish score 		: 提交之后finish的分数
submit like score 			: 提交之后like的分数
notes 						: 备注
```



+ 2019-02-28

|           Item            |                           Setting                            |
| :-----------------------: | :----------------------------------------------------------: |
|           model           |                        xDeepFM_MTL.py                        |
|        batch_size         |                             1024                             |
|          epochs           |                              1                               |
|         optimizer         |                             Adam                             |
|           loss            |                        cross entropy                         |
|       loss_weights        |                          [0.7, 0.3]                          |
|      sparse features      | ['uid', 'user_city', 'item_id', 'author_id', 'item_city','channel','music_id', 'did'] |
|  sparse feature process   |                         LabelEncoder                         |
|      dense features       |                      ['video_duration']                      |
|   dense feature process   |                         MinmaxScaler                         |
|     embedding feature     |                              —                               |
| embedding feature process |                              —                               |
|          target           |                      ['finish', 'like']                      |
|        finish loss        |                              —                               |
|         like loss         |                              —                               |
|      save model file      |                  model_20190228000313.model                  |
|     save result file      |                   result_201902280313.csv                    |
|    submit finish score    |                            0.7061                            |
|     submit like score     |                            0.9228                            |
|           notes           |                              —                               |

