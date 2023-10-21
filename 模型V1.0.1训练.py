import seaborn as sns
import pickle
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import PIL
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from keras import layers
import basic
import models


# 3维卷积，输入格式：(batch_size, frames, height, width, channels)
def build_model_3D(input_shape):
    # 输入层:(100,109,109,3)
    inputs = keras.Input(shape=input_shape)

    # CBS(k=3,s=1):(100,109,109,3)-->(100,109,109,32)
    x = layers.Conv3D(32, 3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # CBS(k=3,s=2):(100,109,109,32)-->(50,55,55,64)
    x = model_CBS(x, 64, 3, 2, 'same')

    # CBS(k=3,s=1):(50,55,55,64)-->(50,55,55,64)
    x = model_CBS(x, 64, 3, 1, 'same')

    # CBS(k=3,s=2):(50,55,55,64)-->(25,28,28,128)
    x = model_CBS(x, 128, 3, 2, 'same')

    # CBS(k=3,s=1):(25,28,28,128)-->(25,28,28,256)
    x = model_CBS(x, 256, 3, 1, 'same')

    # CBS(k=3,s=2):(25,28,28,256)-->(13,14,14,256)
    x = model_CBS(x, 256, 3, 2, 'same')

    # CBS(k=3,s=1):(13,14,14,256)-->(13,14,14,512)
    x = model_CBS(x, 512, 3, 1, 'same')

    # CBS(k=3,s=2):(13,14,14,512)-->(7,7,7,512)
    x = model_CBS(x, 512, 3, 2, 'same')

    # CBS(k=3,s=2):(7,7,7,512)-->(7,7,7,1024)
    x = model_CBS(x, 1024, 3, 1, 'same')

    # 最大池化层:(7,7,7,1024)-->(1,1,1,1024)
    x = layers.MaxPooling3D((7, 7, 7))(x)

    # 展平层：(1,1,1,1024)-->(1024)
    x = layers.Flatten()(x)

    # 全连接层+输出层
    x = layers.Dense(units=512)(x)
    x = layers.Dense(units=256)(x)
    x = layers.Dense(units=128)(x)
    outputs = layers.Dense(units=70)(x)

    # 构建模型
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
    # 可以考虑zeropadding3d层


def train_model(learning_rate, window):
    # 网络编译，训练以及模型保存
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error',
                  metrics='mean_squared_logarithmic_error')
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    model.save(r'model_data/model3D_' + str(window) + '.h5')

    # 获取训练loss数据
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_msle = history_dict['mean_squared_logarithmic_error']  # 训练集的百分比误差
    val_msle = history_dict['val_mean_squared_logarithmic_error']  # 验证集的百分比误差

    # （11）绘制训练损失和验证损失
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')  # 训练集损失
    plt.plot(range(epochs), val_loss, label='val_loss')  # 验证集损失
    plt.plot([-1, 150], [0.02, 0.02], c='black', lw=1, ls='--')  # 临界点直线
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('picture/model_train_effects/loss' + '.jpg')
    plt.show()

    # （12）绘制训练百分比误差和验证百分比误差
    plt.figure()
    plt.plot(range(epochs), train_msle, label='train_msle')  # 训练集指标
    plt.plot(range(epochs), val_msle, label='val_msle')  # 验证集指标
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('msle')
    # plt.savefig('picture/model_train_effects/msle' + '.jpg')
    plt.show()


if __name__ == "__main__":
    # 训练窗口window,预测窗口predict,批处理batch_size，迭代epochs,学习率lr,训练集比例train_per，验证集比例val_per
    # 使用的方法:1.降低学习率 2.加深网络 3.提高dropout 4.增大验证集
    window = 50
    predict = 70
    batch_size = 2
    epochs = 300
    learning_rate = 0.0002
    train_per = 0.8
    val_per = 0.9

    # 读取训练x与训练y数据。
    # basic.getVIT5dimensions_shape_image(window)
    # basic.getY_train()
    # basic.getCycles_IR(window)
    # basic.getCycles_CT(window)
    VIT5 = np.load('./DataNpy/VIT5dimensions_image_' + str(window) + '.npy')
    QD3 = np.load('./DataNpy/QD3dimensions_image.npy')
    IR = np.load("./DataNpy/IR_" + str(window) + ".npy")
    CT = np.load("./DataNpy/CT_" + str(window) + ".npy")
    VIT5, IR, CT, QD3 = VIT5[:140], IR[:140], CT[:140], QD3[:140]

    # 归一化
    QD3_scaled = (QD3 - QD3.min()) / (QD3.max() - QD3.min())
    IR_scaled = (IR - IR.min()) / (IR.max() - IR.min())
    CT_scaled = (CT - CT.min()) / (CT.max() - CT.min())

    # 数据集格式：
    x = VIT5  # (None, window, 109, 109, 3)
    y = QD3_scaled  # (None, 1)
    x1 = IR_scaled  # (None, window)
    x2 = CT_scaled  # (None, window)

    # 构造训练集，验证集和测试集,0-80%训练，80%-90%验证，90%-100%测试
    x_train, y_train, x_val, y_val, x_test, y_test = basic.data_devide(x, y, window, train_per, val_per)

    # 随机打乱,data与label一致
    np.random.seed(5)
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :, :, :, :]
    y_train = y_train[permutation]

    x1_train = x1[:110]
    x2_train = x2[:110]
    x1_val = x1[110:126]
    x2_val = x2[110:126]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, x1_train, x2_train, y_train))
    train_ds = train_ds.batch(batch_size).shuffle(1000)
    # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_ds = train_ds.batch(batch_size).shuffle(1000)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, x1_val, x2_val, y_val))
    val_ds = val_ds.batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size)

    # 取出一个batch的数据 x_train:(None,100,3,109,109) y_train:(None,70)
    # print('x_train.shape:', sample[0].shape)
    # print('y_train.shape:', sample[1].shape)
    sample = next(iter(train_ds))

    # 构造输入层，输入层要和x_train的shape一致,input_shape:(100,3,109,109)
    input_shape = sample[0].shape[1:]
    input_shape1 = sample[1].shape[1:]
    input_shape2 = sample[2].shape[1:]

    # 搭建模型
    model = models.build_model_108(input_shape, input_shape1, input_shape2, window)

    # 训练模型
    train_model(learning_rate, window)

    # 模型预测
    model = tf.keras.models.load_model(r'model_data/model3D_' + str(window) + '.h5')
    ypred = np.zeros((len(x_test), 1))
    for i in range(len(x_test)):
        ypred[i] = model.predict(x_test[i:i + 1])

    # 还原数据真实值(反归一化)
    y_realpredict = (ypred * (QD3.max() - QD3.min())) + QD3.min()
    y_realtest = (y_test * (QD3.max() - QD3.min())) + QD3.min()

    # 评估模型
    model.evaluate(test_ds)
    x_label = np.arange(1, len(x_test) + 1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(x_label, y_realtest, 'b-', label='actual')
    ax.plot(x_label, y_realpredict, 'r--', label='predict')
    ax.set_xticks(x_label[::1])
    ax.set_xlabel('battery')
    ax.set_ylabel('cycles');
    ax.set_title('result')
    plt.legend()
    plt.show()
    print(1)
