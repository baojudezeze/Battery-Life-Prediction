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
from keras.layers import Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

scaler = MinMaxScaler(feature_range=(-1, 1))

files = os.listdir('./Data3')
cellsall_x = np.zeros((1, 100, 4))
cellsall_y = np.zeros((1, 100))

for f in files:
    path = os.path.join('./Data3/' + f)
    print(path)
    bat_dict = pickle.load(open(path, 'rb'))

    cells = np.zeros((len(bat_dict), 100, 4))
    cells_y = np.zeros((len(bat_dict), 100))

    for i in range(len(bat_dict)):
        cycle = np.zeros((4, 100))  # 取前100个循环做训练
        cycle[0] = bat_dict['b1c' + str(i)]['summary']['QC'][:100]
        cycle_y = bat_dict['b1c' + str(i)]['summary']['QC'][:100]
        cycle[1] = bat_dict['b1c' + str(i)]['summary']['IR'][:100]
        cycle[2] = bat_dict['b1c' + str(i)]['summary']['Tavg'][:100]
        cycle[3] = bat_dict['b1c' + str(i)]['summary']['chargetime'][:100]
        cycle = cycle.T  # 结构(100,4)
        cells[i] = cycle  # 结构(48,100,4)
        cells_y[i] = cycle_y

    cellsall_x = np.concatenate((cellsall_x, cells), axis=0)
    cellsall_y = np.concatenate((cellsall_y, cells_y), axis=0)

cellsall_x = cellsall_x[1:]  # (327cells,100,4)
cellsall_y = cellsall_y[1:]  # (327cells,100)

X_slides = []
Y_slides = []

# for k in range(len(cellsall_x)):
#     for i in range(10, 99):
#         X_slides.append(cellsall_x[k][i - 10:i])
#         Y_slides.append(cellsall_y[k][i])
#
for i in range(10, 99):
    X_slides.append(cellsall_x[0][i - 10:i])
    Y_slides.append(cellsall_y[0][i])

X_slides, Y_slides = np.array(X_slides), np.array(Y_slides)

total_num = len(X_slides)  # 一共有多少组序列
train_num = int(total_num * 0.8)  # 前80%的数据用来训练
val_num = int(total_num * 0.9)  # 前80%-90%的数据用来训练验证
# 剩余数据用来测试

batch_size = 32
epochs = 600  # 迭代300次

x_train, y_train = X_slides[:train_num], Y_slides[:train_num]  # 训练集
x_val, y_val = X_slides[train_num: val_num], Y_slides[train_num: val_num]
x_test, y_test = X_slides[val_num:], Y_slides[val_num:]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).shuffle(100000)
# 验证集
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
sample = next(iter(train_ds))  # 取出一个batch的数据
print('x_train.shape:', sample[0].shape)  # (batch_size,10,4)
print('y_train.shape:', sample[1].shape)  # (batch_size,1)

# 输入层要和x_train的shape一致，但注意不要batch维度
input_shape = sample[0].shape[1:]  # [10,4]

# 构造输入层
inputs = keras.Input(shape=input_shape)  # [None, 10, 4]
# x = layers.LSTM(units=200, return_sequences=True)(inputs)
x = layers.GRU(units=200, return_sequences=True)(inputs)
x = layers.Dropout(0.3)(x)
# x = layers.LSTM(units=200, return_sequences=True)(x)
x = layers.GRU(units=200, return_sequences=True)(x)
x = layers.Dropout(0.3)(x)
# x = layers.LSTM(units=200, return_sequences=True)(x)
x = layers.GRU(units=200, return_sequences=True)(x)
x = layers.Dropout(0.3)(x)
# x = layers.LSTM(units=200)(x)
x = layers.GRU(units=200)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(units=1)(x)
outputs = layers.Dense(1)(x)
# 构建模型
model = keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(0.001),  # adam优化器学习率0.001
              loss=tf.keras.losses.MeanAbsoluteError(),  # 标签和预测之间绝对差异的平均值
              metrics=tf.keras.losses.MeanSquaredLogarithmicError())  # 计算标签和预测之间的对数误差均方值。 MeanSquaredError  Logarithmic


# 网络训练, history保存训练时的信息
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

history_dict = history.history  # 获取训练的数据字典
train_loss = history_dict['loss']  # 训练集损失
val_loss = history_dict['val_loss']  # 验证集损失
train_msle = history_dict['mean_squared_logarithmic_error']  # 训练集的百分比误差 _logarithmic_
val_msle = history_dict['val_mean_squared_logarithmic_error']  # 验证集的百分比误差 mean_absolute_error

# （11）绘制训练损失和验证损失
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
plt.plot(range(epochs), val_loss, label='val_loss')  # 验证集损失
plt.legend()  # 显示标签`
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

