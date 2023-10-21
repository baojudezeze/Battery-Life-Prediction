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
X_cells = np.zeros((1, 100))
Y_cells = np.zeros((1, 100))

for f in files:
    path = os.path.join('./Data3/' + f)
    print(path)
    bat_dict = pickle.load(open(path, 'rb'))

    for i in range(len(bat_dict)):
        QC = bat_dict['b1c' + str(i)]['summary']['QC']  # (326,1)
        X_slides = []
        Y_slides = []

        for j in range(100, len(QC) - 1):
            X_slides.append(QC[j-100:j])
            Y_slides.append(QC[j - 99:j + 1])
        X_slides, Y_slides = np.array(X_slides), np.array(Y_slides)
        X_cells = np.concatenate((X_cells, X_slides), axis=0)
        Y_cells = np.concatenate((Y_cells, Y_slides), axis=0)
    X_cells = np.concatenate((X_cells, X_cells), axis=0)
    Y_cells = np.concatenate((Y_cells, Y_cells), axis=0)

X_cellsALL = X_cells[1:]  # (341460,100)
Y_cellsALL = Y_cells[1:]  # (341460,100)

total_num = len(X_cellsALL)  # 一共有多少组序列
train_num = int(total_num * 0.8)  # 前80%的数据用来训练
val_num = int(total_num * 0.9)  # 前80%-90%的数据用来训练验证
# 剩余数据用来测试

batch_size = 1024
epochs = 3  # 迭代300次

x_train, y_train = X_cellsALL[:train_num], Y_cellsALL[:train_num]  # 训练集
x_val, y_val = X_cellsALL[train_num: val_num], Y_cellsALL[train_num: val_num]
x_test, y_test = X_cellsALL[val_num:], Y_cellsALL[val_num:]


# 环境变量的配置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
model = tf.keras.models.load_model(r'model_data/model.h5')

# a = x_test[0].reshape(1, 100)
# y_pred = model.predict(a)
# for i in range(10):
#     y_pred = model.predict(y_pred)
#     # test = np.concatenate((test, y_pred[-1]), axis=0)
# print(y_pred)
# print(a)

window = 500

bat_dict_test = pickle.load(open('./Data/2019-01-24.pkl', 'rb'))
allx = bat_dict_test['b1c1']['summary']['QC'][:window]
test = bat_dict_test['b1c1']['summary']['QC'][:100]
test = test.reshape(1, 100)
y_pred = model.predict(test)
print(y_pred)
pd = np.hstack((test, y_pred[0, ].reshape(1, 100)))

for i in range(3):
    y_pred = model.predict(y_pred)
    pd = np.hstack((pd, y_pred[0, ].reshape(1, 100)))


pd = pd.reshape((window, 1))
allx = allx.reshape((window, 1))

# （12）绘制训练百分比误差和验证百分比误差
plt.figure()
plt.plot(range(window), pd, label='pd')  # 训练集指标
plt.plot(range(window), allx, label='allx')  # 验证集指标
plt.legend()  # 显示标签
plt.xlabel('cycles')
plt.ylabel('QC')
plt.show()

