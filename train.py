import tensorflow as tf
import pickle
import numpy as np
from keras import layers
from tensorflow import keras
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 环境变量的配置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def build_sequences(text, window_size):
    # text:list of capacity
    x, y = [], []
    for i in range(len(text) - window_size):
        sequence = text[i:i + window_size]
        target = text[i + 1:i + 1 + window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size + 1], data_sequence[window_size + 1:]
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)


windows = 128

#  加载MIT数据集
files = os.listdir('./Data3')
X_cellsALL = np.zeros((1, windows))
Y_cellsALL = np.zeros((1, windows))
for f in files:
    path = os.path.join('./Data3/' + f)
    print(path)
    bat_dict = pickle.load(open(path, 'rb'))
    X_cells = np.zeros((1, windows))
    Y_cells = np.zeros((1, windows))

    for i in range(len(bat_dict)):  # len(bat_dict)
        QC = bat_dict['b1c' + str(i)]['summary']['QC']  # (326,1)

        X_slides = []
        Y_slides = []

        for j in range(len(QC) - windows):  # len(QC) - 100
            X_slides.append(QC[j:j + windows])
            Y_slides.append(QC[j + 1:j + windows + 1])
        X_slides, Y_slides = np.array(X_slides), np.array(Y_slides)
        X_cells = np.concatenate((X_cells, X_slides), axis=0)
        Y_cells = np.concatenate((Y_cells, Y_slides), axis=0)

    X_cells = X_cells[1:]  # (341460,100)
    Y_cells = Y_cells[1:]  # (341460,100)

    X_cellsALL = np.concatenate((X_cellsALL, X_cells), axis=0)
    Y_cellsALL = np.concatenate((Y_cellsALL, Y_cells), axis=0)

X_cellsALL = X_cellsALL[1:]  # (341460,100)
Y_cellsALL = Y_cellsALL[1:]  # (341460,100)
print('data load success.')

# test
Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('CALCE.npy', allow_pickle=True)
Battery = Battery.item()
name = Battery_list[0]
fea = 128
train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size=fea)

X_cellsALL = train_x
Y_cellsALL = train_y

total_num = len(X_cellsALL)  # 一共有多少组序列
train_num = int(total_num * 0.8)  # 前80%的数据用来训练
val_num = int(total_num * 0.9)  # 前80%-90%的数据用来训练验证
# 剩余数据用来测试

batch_size = 32
epochs = 500  # 迭代200次

x_train, y_train = X_cellsALL[:train_num], Y_cellsALL[:train_num]  # 训练集
x_val, y_val = X_cellsALL[train_num: val_num], Y_cellsALL[train_num: val_num]
x_test, y_test = X_cellsALL[val_num:], Y_cellsALL[val_num:]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).shuffle(1000000)
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
inputs = keras.Input(shape=input_shape)  # [None, 100]
x = layers.Reshape(target_shape=(inputs.shape[1], 1))(inputs)
# x = layers.LSTM(units=100, return_sequences=True)(x)
# x = layers.Dropout(0.3)(x)
# x = layers.LSTM(units=100, return_sequences=True)(x)
# x = layers.Dropout(0.3)(x)
x = layers.LSTM(units=256)(x)
# x = layers.Dropout(0.3)(x)
outputs = layers.Dense(units=128)(x)
# 构建模型
model = keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(0.001),  # adam优化器学习率0.001
              loss='mean_squared_error',  # 标签和预测之间绝对差异的平均值
              metrics='mean_squared_logarithmic_error')  # 计算标签和预测之间的对数误差均方值。
# 网络训练, history保存训练时的信息
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# 保存模型
model.save(r'model_data/model.h5')

history_dict = history.history  # 获取训练的数据字典
train_loss = history_dict['loss']  # 训练集损失
val_loss = history_dict['val_loss']  # 验证集损失
train_msle = history_dict['mean_squared_logarithmic_error']  # 训练集的百分比误差
val_msle = history_dict['val_mean_squared_logarithmic_error']  # 验证集的百分比误差
print(train_loss)
#

#
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
