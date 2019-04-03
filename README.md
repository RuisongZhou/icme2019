# README

+ `config.py`：模型配置文件，参数在该文件中进行改动。
+ `train.py`：模型训练、训练预测在该文件中。
+ `dataloader.py`：数据的读取、特征合并、数据预处理、产生特征列表在该文件中。
+ `model/`：模型文件存在该文件夹下。
+ `result/`：预测的结果文件保存在该文件夹下。
+ `source/models/`：训练后的模型文件保存在该文件夹下。
+ `source/transformer/`：数据处理过程的LabelEncoder和MinmaxScaler二进制文件。

## 模型输入

+ 模型输入包含以下几个模块
  + 读取模型文件：该代码在`conf.train=True`模式下只会读取训练集，不会读取测试集。在`conf.train=False`模式下会读取训练集和测试集，并统一编码和归一化。
  + 特征合并：如果还使用face特征以及title特征，那么会在训练集和测试集中添加特征列，这些特征列的值与`item_id`一一对应。
  + 预处理：预处理包括两部分，对sparse feature的空值填充-1，对dense feature的空值填充0。还有对sparse feature进行统计并转化成index，对dense feature进行归一化到(0,1)之间。
  + 产生sparse/dense_feature_list：列表元素为SingleFeat实例，SingleFeat为namedtuple，用于之后输入模型来构建模型(模型构建需要知道sparse feature有多少种值)。
+ 模型输入格式
  + 模型的输入为`np.array`的`list`，每一个`np.array`表示一种特征。
  + 输入`list`中sparse feature的输入在前，dense feature输入在后。两者的feature输入顺序要和模型构建时传递进去的sparse feature和dense feature列表的顺序一致。

## 模型训练预测

+ 模型训练包括两步
  + 构建模型：通过sparse feature和dense feature的个数，embedding维度等构建模型。
  + 训练模型：baseline使用`tf.keras.Model()`来训练预测模型，也可以自定义模型放在`model/`文件夹下。
+ 模型预测
  + 在`conf.train=False`模式下模型训练完之后会自动地进行预测，并保存`result_(TimeStamp).csv`文件到`result/`文件夹下。

## 模型训练预测记录表

+ 所有模型训练、预测和提交之后填写模型训练预测记录表，方便进行模型比较和改进。
+ 模型训练预测记录表见`RECORD.md`。