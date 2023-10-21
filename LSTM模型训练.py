import tensorflow as tf
import pickle
import numpy as np
from keras import layers
from tensorflow import keras
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd

files = os.listdir('./Data_VIT100')
cellsall_x = np.zeros((1, 150, 110, 4))
cellsall_y = np.zeros((1, 1))


for f in files:
    path = os.path.join('./Data_VIT100/' + f)
    print(path)
    bat_dict = pickle.load(open(path, 'rb'))

    VIT = np.zeros((150, 110, 4))
    cells_x = np.zeros((len(bat_dict), 150, 110, 4))
    cells_y = np.zeros((len(bat_dict), 1))
    for j in range(len(bat_dict)):
        for i in range(150):
            cycle = {}
            cycle = bat_dict['b1c' + str(j)][str(i)]  # 第一块电池的第i个循环的所有VIT数据
            cycle = pd.DataFrame(cycle)
            cycle = np.array(cycle)
            scaler = MinMaxScaler(feature_range=(0, 1))
            cycle = scaler.fit_transform(cycle)
            VIT[i] = cycle
        VIT = VIT.astype('float32')
        cells_x[j] = VIT
        cells_y[j] = len(bat_dict['b1c' + str(j)])
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # cells_y = scaler.fit_transform(cells_y)

    cellsall_x = np.concatenate((cellsall_x, cells_x), axis=0)
    cellsall_y = np.concatenate((cellsall_y, cells_y), axis=0)

cellsall_x = cellsall_x[1:]
cellsall_y = cellsall_y[1:]
cellsall_x = cellsall_x.astype('float32')  # 结构(48+46+46, 100, 110, 4)
cellsall_y = cellsall_y.astype('float32')  # 结构(48+46+46,1)

# 划分训练集
total_num = len(cellsall_x)  # 一共有多少组序列
train_num = int(total_num * 0.8)  # 前80%的数据用来训练
val_num = int(total_num * 0.9)  # 前80%-90%的数据用来训练验证
# 剩余数据用来测试

x_train, y_train = cellsall_x[:train_num], cellsall_y[:train_num]  # 训练集
x_val, y_val = cellsall_x[train_num: val_num], cellsall_y[train_num: val_num]  # 验证集
x_test, y_test = cellsall_x[val_num:], cellsall_y[val_num:]  # 测试集

# 构造数据集
batch_size = 2  # 每次迭代处理128个序列
# 训练集
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).shuffle(100000)
# 验证集
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(batch_size)
# 测试集
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
sample = next(iter(train_ds))  # 取出一个batch的数据
print('x_train.shape:', sample[0].shape)  # (batch_size,100,110,4)
print('y_train.shape:', sample[1].shape)  # (batch_size,1)

# 模型搭建
# 输入层要和x_train的shape一致，但注意不要batch维度
input_shape = sample[0].shape[1:]  # [100,110,4]

# 构造输入层
inputs = keras.Input(shape=input_shape)  # [None, 10, 12]

# 调整维度 [None,10,12]==>[None,10,12,1]
# x = layers.Reshape(target_shape=(inputs.shape[1], inputs.shape[2], 1))(inputs)

# 卷积+BN+Relu  [None,100,110,4]==>[None,100,110,1]
x = layers.Conv2D(1, kernel_size=(3, 3), strides=1, padding='same', use_bias=False,
                  kernel_regularizer=keras.regularizers.l2(0.01))(inputs)

x = layers.BatchNormalization()(x)  # 批标准化
x = layers.Activation('relu')(x)  # relu激活函数

# 池化下采样 [None,100,110,1]==>[None,100,55,1]
x = layers.MaxPool2D(pool_size=(1, 2))(x)

# 把最后一个维度挤压掉 [None,100,55,1]==>[None,100,55]
x = tf.squeeze(x, axis=-1)

# 1*1卷积调整通道数 [None,100,55]==>[None,100,1]
x = layers.Conv1D(1, 1, strides=1, padding='same', use_bias=False,
                  kernel_regularizer=keras.regularizers.l2(0.01))(x)
# [None,100,1]==>[None,50,1]
x = layers.MaxPool1D(pool_size=2)(x)
# [None,50,1]==>[None,50]
x = tf.squeeze(x, axis=-1)

# [None,10,6] ==> [None,10,16]
# 第一个LSTM层, 如果下一层还是LSTM层就需要return_sequences=True, 否则就是False
# x = layers.LSTM(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

x = layers.Dropout(0.2)(x)  # 随机杀死神经元防止过拟合

# 输出层 [None,16]==>[None,1]
outputs = layers.Dense(1)(x)

# 构建模型
model = keras.Model(inputs, outputs)

# 查看模型架构
model.summary()

# 编译模型
# 网络编译
model.compile(optimizer=keras.optimizers.Adam(0.001),  # adam优化器学习率0.001
              loss=tf.keras.losses.MeanAbsoluteError(),  # 标签和预测之间绝对差异的平均值
              metrics=tf.keras.losses.MeanSquaredLogarithmicError())  # 计算标签和预测之间的对数误差均方值。

epochs = 600  # 迭代300次

# 网络训练, history保存训练时的信息
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

history_dict = history.history  # 获取训练的数据字典
train_loss = history_dict['loss']  # 训练集损失
val_loss = history_dict['val_loss']  # 验证集损失
train_msle = history_dict['mean_squared_logarithmic_error']  # 训练集的百分比误差
val_msle = history_dict['val_mean_squared_logarithmic_error']  # 验证集的百分比误差

# （11）绘制训练损失和验证损失
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
plt.plot(range(epochs), val_loss, label='val_loss')  # 验证集损失
plt.legend()  # 显示标签
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# （12）绘制训练百分比误差和验证百分比误差
plt.figure()
plt.plot(range(epochs), train_msle, label='train_msle')  # 训练集指标
plt.plot(range(epochs), val_msle, label='val_msle')  # 验证集指标
plt.legend()  # 显示标签
plt.xlabel('epochs')
plt.ylabel('msle')
plt.show()

y_pred = model.predict(x_test)
for i in range(len(y_pred)):
    print(y_pred[i], y_test[i])

# # 对整个测试集评估
# model.evaluate(test_ds)
#
# # 预测
# y_pred = model.predict(x_test)

# # 获取标签值对应的时间
# df_time = times[-len(y_test):]
#
# # 绘制对比曲线
# fig = plt.figure(figsize=(10, 5))  # 画板大小
# ax = fig.add_subplot(111)  # 画板上添加一张图
# # 绘制真实值曲线
# ax.plot(df_time, y_test, 'b-', label='actual')
# # 绘制预测值曲线
# ax.plot(df_time, y_pred, 'r--', label='predict')
# # 设置x轴刻度
# ax.set_xticks(df_time[::7])
#
# # 设置xy轴标签和title标题
# ax.set_xlabel('Date')
# ax.set_ylabel('Temperature');
# ax.set_title('result')
# plt.legend()
# plt.show()
