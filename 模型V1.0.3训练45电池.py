import csv

import numpy as np
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from tensorflow import keras

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
    x = models.model_CBS(x, 64, 3, 2, 'same')

    # CBS(k=3,s=1):(50,55,55,64)-->(50,55,55,64)
    x = models.model_CBS(x, 64, 3, 1, 'same')

    # CBS(k=3,s=2):(50,55,55,64)-->(25,28,28,128)
    x = models.model_CBS(x, 128, 3, 2, 'same')

    # CBS(k=3,s=1):(25,28,28,128)-->(25,28,28,256)
    x = models.model_CBS(x, 256, 3, 1, 'same')

    # CBS(k=3,s=2):(25,28,28,256)-->(13,14,14,256)
    x = models.model_CBS(x, 256, 3, 2, 'same')

    # CBS(k=3,s=1):(13,14,14,256)-->(13,14,14,512)
    x = models.model_CBS(x, 512, 3, 1, 'same')

    # CBS(k=3,s=2):(13,14,14,512)-->(7,7,7,512)
    x = models.model_CBS(x, 512, 3, 2, 'same')

    # CBS(k=3,s=2):(7,7,7,512)-->(7,7,7,1024)
    x = models.model_CBS(x, 1024, 3, 1, 'same')

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
    # 获取训练loss数据
    history_dict = history.history  # 获取训练的数据字典
    train_loss = history_dict['loss']  # 训练集损失
    val_loss = history_dict['val_loss']  # 验证集损失
    train_msle = history_dict[losstype]  # 训练集的百分比误差mean_squared_logarithmic_error
    val_msle = history_dict['val_' + losstype]  # 验证集的百分比误差

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

    with open('./train_loss.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(train_loss))

    with open('./val_loss.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(val_loss))

    # （12）绘制训练百分比误差和验证百分比误差
    plt.figure()
    plt.plot(range(epochs), train_msle, label='train_msle')  # 训练集指标
    plt.plot(range(epochs), val_msle, label='val_msle')  # 验证集指标
    plt.legend()  # 显示标签
    plt.xlabel('epochs')
    plt.ylabel('msle')
    # plt.savefig('picture/model_train_effects/msle' + '.jpg')
    plt.show()


def scaled(q):
    q = (q - q.min()) / (q.max() - q.min())
    return q


if __name__ == "__main__":
    # 训练窗口window,预测窗口predict,批处理batch_size，迭代epochs,学习率lr,训练集比例train_per，验证集比例val_per
    # 使用的方法:1.降低学习率 2.加深网络 3.提高dropout 4.增大验证集
    window = 50
    predict = 70
    batch_size = 10
    epochs = 300
    learning_rate = 0.00001
    train_per = 0.6
    val_per = 0.99
    losstype = 'mean_squared_logarithmic_error'
    # 'mean_squared_logarithmic_error'
    # 'mean_squared_error'
    # 'mean_absolute_error'

    # 读取训练x与训练y数据。
    # basic.getVIT5dimensions_shape_image(window)
    # basic.getY_train()
    # basic.getCycles_IR(window)
    # basic.getCycles_CT(window)
    # basic.get_IC(window)
    # basic.get_QV50(window)
    # basic.get_WSD_IC(window)
    # basic.get_WSD_QV50(window)
    # basic.get_PICC(window)
    # basic.get_Qd(window)
    VIT5 = np.load('./DataNpy/VIT5dimensions_image_' + str(window) + '.npy')

    # y更换为循环寿命而不是SOH
    with open("./Data/cyclelife2.csv", "r") as f:
        csv_reader_lines = csv.reader(f)  # 用csv.reader读文件
        date_PyList = []
        for one_line in csv_reader_lines:
            date_PyList.append(one_line)  # 逐行将读到的文件存入python的列表
        date_ndarray = np.array(date_PyList)  # 将python列表转化为ndarray
    QD3 = date_ndarray.astype(np.float64)
    # QD3 = np.load('./DataNpy/QD3dimensions_image.npy')

    IR = np.load("./DataNpy/IR.npy")
    CT = np.load("./DataNpy/CT.npy")
    Qd = np.load('./DataNpy/Qd.npy')

    # 其他HI
    IC = np.load('./DataNpy/IC.npy')
    QV50 = np.load('./DataNpy/QV50.npy')
    WSD_IC = np.load('./DataNpy/WSD_IC.npy')
    WSD_QV50 = np.load('./DataNpy/WSD_QV50.npy')
    PICC = np.load('./DataNpy/PICC.npy')

    # 暂时取前3个数据集的数据
    VIT5, IR, CT, QD3, WSD_IC, WSD_QV50, PICC = VIT5[140:], IR[140:], CT[140:], QD3, WSD_IC[140:], \
                                                WSD_QV50[140:], PICC[140:]

    # 归一化
    QD3_scaled = scaled(QD3)
    Qd_scaled = scaled(Qd)
    IR_scaled = scaled(IR)
    CT_scaled = scaled(CT)
    WSD_IC_scaled = scaled(WSD_IC)
    WSD_QV50_scaled = scaled(WSD_QV50)
    PICC_scaled = scaled(PICC)

    # 数据集格式：
    x = VIT5  # (None, window, 109, 109, 3)
    y = QD3_scaled  # (None, 1)
    a = WSD_IC_scaled  # (None, window)
    b = WSD_QV50_scaled  # (None, window)
    c = IR_scaled  # (None, window)

    # 构造训练集，验证集和测试集,0-80%训练，80%-90%验证，90%-100%测试
    x_train, y_train, x_val, y_val, x_test, y_test, a_train, b_train, c_train, a_val, b_val, c_val, a_test, b_test, c_test \
        = basic.data_devide(x, y, a, b, c, window, train_per, val_per)

    # 随机打乱,data与label一致
    np.random.seed(5)
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :, :, :, :]
    y_train = y_train[permutation]
    a_train = a_train[permutation, :]
    b_train = b_train[permutation, :]
    c_train = c_train[permutation, :]

    permutation_val = np.random.permutation(x_val.shape[0])
    x_val = x_val[permutation_val, :, :, :, :]
    y_val = y_val[permutation_val]
    a_val = a_val[permutation_val, :]
    b_val = b_val[permutation_val, :]
    c_val = c_val[permutation_val, :]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, a_train, b_train, c_train))
    train_ds = train_ds.batch(batch_size)

    # 取出一个batch的数据 x_train:(None,100,3,109,109) y_train:(None,70)
    # print('x_train.shape:', sample[0].shape)
    # print('y_train.shape:', sample[1].shape)
    sample = next(iter(train_ds))

    # 构造输入层，输入层要和x_train的shape一致,input_shape:(100,3,109,109)
    input_shape = sample[0].shape[1:]
    input_shape_a = sample[1].shape[1:]
    input_shape_b = sample[2].shape[1:]
    input_shape_c = sample[3].shape[1:]

    # 搭建模型
    # model = models.build_model_200(input_shape, input_shape_a, input_shape_b, input_shape_c, window)
    #
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=losstype,
    #               metrics=losstype)  # mean_squared_logarithmic_error
    # history = model.fit([x_train, a_train, b_train, c_train], y_train, batch_size=batch_size, epochs=epochs,
    #                     validation_data=[[x_val, a_val, b_val, c_val], y_val])
    # model.save(r'model_data/model3D_' + str(window) + '.h5')
    # train_model(learning_rate, window)

    # 模型预测
    model = tf.keras.models.load_model(r'model_data/model3D_' + str(window) + '.h5')
    ypred = np.zeros((len(x_val), 1))
    # for i in range(len(x_test)):
    #     ypred[i] = model.predict([x_test[i:i + 1], a_test[i:i + 1], b_test[i:i + 1], c_test[i:i + 1]])

    for i in range(len(x_val)):
        ypred[i] = model.predict([x_val[i:i + 1], a_val[i:i + 1], b_val[i:i + 1], c_val[i:i + 1]])

    # 还原数据真实值(反归一化)
    y_realpredict = (ypred * (QD3.max() - QD3.min())) + QD3.min()
    y_realtest = (y_val * (QD3.max() - QD3.min())) + QD3.min()

    a = []
    for i in range(len(y_realpredict)):
        a.append((y_val[i] - ypred[i]) / y_val[i])

    with open('./predict.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(y_realpredict))

    # 评估模型
    # model.evaluate(test_ds)
    x_label = np.arange(1, len(x_val) + 1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(x_label, y_realtest, 'b-', label='actual')
    ax.plot(x_label, y_realpredict, 'r--', label='predict')
    ax.set_xticks(x_label[::1])
    ax.set_xlabel('battery')
    ax.set_ylabel('cycles')
    ax.set_title('result')
    plt.legend()
    plt.show()
    print(1)
