import numpy
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

files = os.listdir('./Data3')

X_cellsALL = np.zeros((1, 1))
Y_cellsALL = np.zeros((1, 1))

for f in files:
    path = os.path.join('./Data3/' + f)
    print(path)
    bat_dict = pickle.load(open(path, 'rb'))
    X_cells = np.zeros((1, 1))
    Y_cells = np.zeros((1, 1))

    for i in range(len(bat_dict)):
        QC = bat_dict['b1c' + str(i)]['summary']['QC']  # (326,1)
        X_slides = []
        Y_slides = []

        for j in range(len(QC) - 1):  # len(QC) - 100
            X_slides.append(QC[j])
            Y_slides.append(QC[j + 1])
        X_slides, Y_slides = np.array(X_slides), np.array(Y_slides)
        X_cells = np.hstack((X_cells, X_slides.reshape((1, -1))))
        Y_cells = np.hstack((Y_cells, Y_slides.reshape((1, -1))))

    X_cells = X_cells[0, 1:]  # (341460,1)
    Y_cells = Y_cells[0, 1:]  # (341460,1)

    X_cellsALL = np.hstack((X_cellsALL, X_cells.reshape((1, -1))))
    Y_cellsALL = np.hstack((Y_cellsALL, Y_cells.reshape((1, -1))))

X_cellsALL = X_cellsALL[0, 1:]  # (341460,1)
Y_cellsALL = Y_cellsALL[0, 1:]  # (341460,1)

scaler = MinMaxScaler(feature_range=(0, 1))

X_cellsALL = scaler.fit_transform(X_cellsALL.reshape((-1, 1)))
Y_cellsALL = scaler.fit_transform(Y_cellsALL.reshape((-1, 1)))

total_num = len(X_cellsALL)  # 一共有多少组序列
train_num = int(total_num * 0.8)  # 前80%的数据用来训练
val_num = int(total_num * 0.9)  # 前80%-90%的数据用来训练验证
# 剩余数据用来测试

batch_size = 512
epochs = 300  # 迭代200次

x_train, y_train = X_cellsALL[:train_num], Y_cellsALL[:train_num]  # 训练集
x_val, y_val = X_cellsALL[train_num: val_num], Y_cellsALL[train_num: val_num]
x_test, y_test = X_cellsALL[val_num:], Y_cellsALL[val_num:]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size).shuffle(1000000)
# 验证集
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# 环境变量的配置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
model = tf.keras.models.load_model(r'model_data/model.h5')


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


# test
Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('CALCE.npy', allow_pickle=True)
Battery = Battery.item()
name = Battery_list[0]
feature_size = 128
train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size=feature_size)

window = 800
bat_dict_test = pickle.load(open('./Data/2019-01-24.pkl', 'rb'))
# allx = bat_dict_test['b1c0']['summary']['QC'][:window]

# test1 = bat_dict_test['b1c0']['summary']['QC'][:window]
# test1 = test1.reshape((-1, 1))
test_data = numpy.array(test_data).reshape((1, -1))
inputx = numpy.array(train_data)
inputx = inputx.reshape((1, -1))
inputx = inputx[0, 1:]
inputx = inputx.reshape(1, 128)
y_pred = model.predict(inputx)
pd = np.hstack((inputx, y_pred[0, -1].reshape((1, 1))))
allx = np.hstack((inputx, test_data))
allx = allx[0, :800]
#  以源数据预测
for i in range(window-129):
    y_pred = model.predict(y_pred)
    pd = np.hstack((pd, y_pred[0, -1].reshape((1, 1))))

#  以预测数据预测
# for i in range(len(test1)-1):
#     inputx = numpy.array(y_pred[i, 0]).reshape((-1, 1))
#     y_pred[i+1, 0] = model.predict(inputx)

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
