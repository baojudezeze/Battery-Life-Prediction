import numpy as np
import os
import pickle
import scipy.stats


# 生成RP图函数
def RP(X):
    # normalization to [0,1]
    X = (X - np.max(X)) / (np.max(X) + np.min(X))
    Xlen = X.shape[1]
    # convert to the phase space(第一元素是此时高度，第二个给元素为下一时刻的高度)
    S = np.zeros([Xlen - 1, 2])
    S[:, 0] = X[0, :-1]
    S[:, 1] = X[0, 1:]
    # compute RRP matrix
    R = np.zeros([Xlen - 1, Xlen - 1])
    for i in range(Xlen - 1):
        for j in range(Xlen - 1):
            R[i, j] = sum(pow(S[i, :] - S[j, :], 2))
    # normalization to [0,1] of RP
    R = (R - R.min()) / (R.max() - R.min()) * 1
    # show the heatmap(bwr,coolwarm,GnBu)
    # sns.heatmap(R, cbar=True, square=True, cmap='GnBu', xticklabels=False, yticklabels=False, center=0)
    return R


# 寻找最近点函数
def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    # return a.flat[idx], idx
    return idx


def filter_limit(x, limit):
    for i in range(len(x) - 1):
        if abs(x[i + 1] - x[i]) >= limit:
            x[i + 1] = x[i]
    return x


def mean_filter(input, per):
    output = np.zeros((len(input)))
    for i in range(per, len(input) - per):
        a = sum(input[i - per:i + per])
        mean = a / (2 * per)
        input[i] = mean
    return input


# 根据难度划分数据集
def data_devide(x, y, a, b, c, window, train_per, val_per):
    x1, x2, x3, x4, x5 = np.zeros((1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros(
        (1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3))
    y1, y2, y3, y4, y5 = [], [], [], [], []
    a1, a2, a3, a4, a5 = np.zeros((1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros((1, window))
    b1, b2, b3, b4, b5 = np.zeros((1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros((1, window))
    c1, c2, c3, c4, c5 = np.zeros((1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros((1, window))
    for i in range(len(x)):
        x_dev = x[i].reshape((1, window, 109, 109, 3))
        y_dev = y[i].tolist()
        a_dev = a[i].reshape((1, window))
        b_dev = b[i].reshape((1, window))
        c_dev = c[i].reshape((1, window))
        if 0 <= y[i] < 0.2:
            x1 = np.vstack((x1, x_dev))
            y1.append(y_dev)
            a1 = np.vstack((a1, a_dev))
            b1 = np.vstack((b1, b_dev))
            c1 = np.vstack((c1, c_dev))
        elif 0.2 <= y[i] < 0.4:
            x2 = np.vstack((x2, x_dev))
            y2.append(y_dev)
            a2 = np.vstack((a2, a_dev))
            b2 = np.vstack((b2, b_dev))
            c2 = np.vstack((c2, c_dev))
        elif 0.4 <= y[i] < 0.6:
            x3 = np.vstack((x3, x_dev))
            y3.append(y_dev)
            a3 = np.vstack((a3, a_dev))
            b3 = np.vstack((b3, b_dev))
            c3 = np.vstack((c3, c_dev))
        elif 0.6 <= y[i] < 0.8:
            x4 = np.vstack((x4, x_dev))
            y4.append(y_dev)
            a4 = np.vstack((a4, a_dev))
            b4 = np.vstack((b4, b_dev))
            c4 = np.vstack((c4, c_dev))
        else:
            x5 = np.vstack((x5, x_dev))
            y5.append(y_dev)
            a5 = np.vstack((a5, a_dev))
            b5 = np.vstack((b5, b_dev))
            c5 = np.vstack((c5, c_dev))
    x1, x2, x3, x4, x5 = x1[1:], x2[1:], x3[1:], x4[1:], x5[1:]
    a1, a2, a3, a4, a5 = a1[1:], a2[1:], a3[1:], a4[1:], a5[1:]
    b1, b2, b3, b4, b5 = b1[1:], b2[1:], b3[1:], b4[1:], b5[1:]
    c1, c2, c3, c4, c5 = c1[1:], c2[1:], c3[1:], c4[1:], c5[1:]

    # 拼成训练集
    x_train = np.zeros((1, window, 109, 109, 3))
    y_train = np.zeros((1, 1))
    x_val = np.zeros((1, window, 109, 109, 3))
    y_val = np.zeros((1, 1))
    x_test = np.zeros((1, window, 109, 109, 3))
    y_test = np.zeros((1, 1))

    x_train = np.vstack((x_train, x1[:int(len(x1) * train_per)]))
    x_train = np.vstack((x_train, x2[:int(len(x2) * train_per)]))
    x_train = np.vstack((x_train, x3[:int(len(x3) * train_per)]))
    x_train = np.vstack((x_train, x4[:int(len(x4) * train_per)]))
    x_train = np.vstack((x_train, x5[:int(len(x5) * train_per)]))
    x_train = x_train[1:]
    x_val = np.vstack((x_val, x1[int(len(x1) * train_per):int(len(x1) * val_per)]))
    x_val = np.vstack((x_val, x2[int(len(x2) * train_per):int(len(x2) * val_per)]))
    x_val = np.vstack((x_val, x3[int(len(x3) * train_per):int(len(x3) * val_per)]))
    x_val = np.vstack((x_val, x4[-2].reshape(1, window, 109, 109, 3)))
    x_val = np.vstack((x_val, x5[-2].reshape(1, window, 109, 109, 3)))
    x_val = x_val[1:]
    x_test = np.vstack((x_test, x1[int(len(x1) * val_per):]))
    x_test = np.vstack((x_test, x2[int(len(x2) * val_per):]))
    x_test = np.vstack((x_test, x3[int(len(x3) * val_per):]))
    x_test = np.vstack((x_test, x4[-1].reshape(1, window, 109, 109, 3)))
    x_test = np.vstack((x_test, x5[-1].reshape(1, window, 109, 109, 3)))
    x_test = x_test[1:]

    y_train = np.vstack((y_train, np.array(y1[:int(len(y1) * train_per)])))
    y_train = np.vstack((y_train, np.array(y2[:int(len(y2) * train_per)])))
    y_train = np.vstack((y_train, np.array(y3[:int(len(y3) * train_per)])))
    y_train = np.vstack((y_train, np.array(y4[:int(len(y4) * train_per)])))
    y_train = np.vstack((y_train, np.array(y5[:int(len(y5) * train_per)])))
    y_train = y_train[1:]
    y_val = np.vstack((y_val, np.array(y1[int(len(y1) * train_per):int(len(y1) * val_per)])))
    y_val = np.vstack((y_val, np.array(y2[int(len(y2) * train_per):int(len(y2) * val_per)])))
    y_val = np.vstack((y_val, np.array(y3[int(len(y3) * train_per):int(len(y3) * val_per)])))
    y_val = np.vstack((y_val, np.array(y4[-2]).reshape(1, 1)))
    y_val = np.vstack((y_val, np.array(y5[-2]).reshape(1, 1)))
    y_val = y_val[1:]
    y_test = np.vstack((y_test, np.array(y1[int(len(y1) * val_per):])))
    y_test = np.vstack((y_test, np.array(y2[int(len(y2) * val_per):])))
    y_test = np.vstack((y_test, np.array(y3[int(len(y3) * val_per):])))
    y_test = np.vstack((y_test, np.array(y4[-1]).reshape(1, 1)))
    y_test = np.vstack((y_test, np.array(y5[-1]).reshape(1, 1)))
    y_test = y_test[1:]

    # 附加参数
    a_train = np.zeros((1, window))
    b_train = np.zeros((1, window))
    c_train = np.zeros((1, window))
    a_val = np.zeros((1, window))
    b_val = np.zeros((1, window))
    c_val = np.zeros((1, window))
    a_test = np.zeros((1, window))
    b_test = np.zeros((1, window))
    c_test = np.zeros((1, window))
    a_train = np.vstack((a_train, a1[:int(len(a1) * train_per)]))
    a_train = np.vstack((a_train, a2[:int(len(a2) * train_per)]))
    a_train = np.vstack((a_train, a3[:int(len(a3) * train_per)]))
    a_train = np.vstack((a_train, a4[:int(len(a4) * train_per)]))
    a_train = np.vstack((a_train, a5[:int(len(a5) * train_per)]))
    a_train = a_train[1:]
    b_train = np.vstack((b_train, b1[:int(len(b1) * train_per)]))
    b_train = np.vstack((b_train, b2[:int(len(b2) * train_per)]))
    b_train = np.vstack((b_train, b3[:int(len(b3) * train_per)]))
    b_train = np.vstack((b_train, b4[:int(len(b4) * train_per)]))
    b_train = np.vstack((b_train, b5[:int(len(b5) * train_per)]))
    b_train = b_train[1:]
    c_train = np.vstack((c_train, c1[:int(len(c1) * train_per)]))
    c_train = np.vstack((c_train, c2[:int(len(c2) * train_per)]))
    c_train = np.vstack((c_train, c3[:int(len(c3) * train_per)]))
    c_train = np.vstack((c_train, c4[:int(len(c4) * train_per)]))
    c_train = np.vstack((c_train, c5[:int(len(c5) * train_per)]))
    c_train = c_train[1:]
    a_val = np.vstack((a_val, a1[int(len(a1) * train_per):int(len(a1) * val_per)]))
    a_val = np.vstack((a_val, a2[int(len(a2) * train_per):int(len(a2) * val_per)]))
    a_val = np.vstack((a_val, a3[int(len(a3) * train_per):int(len(a3) * val_per)]))
    a_val = np.vstack((a_val, a4[-2].reshape(1, window)))
    a_val = np.vstack((a_val, a5[-2].reshape(1, window)))
    a_val = a_val[1:]
    b_val = np.vstack((b_val, b1[int(len(b1) * train_per):int(len(b1) * val_per)]))
    b_val = np.vstack((b_val, b2[int(len(b2) * train_per):int(len(b2) * val_per)]))
    b_val = np.vstack((b_val, b3[int(len(b3) * train_per):int(len(b3) * val_per)]))
    b_val = np.vstack((b_val, b4[-2].reshape(1, window)))
    b_val = np.vstack((b_val, b5[-2].reshape(1, window)))
    b_val = b_val[1:]
    c_val = np.vstack((c_val, c1[int(len(c1) * train_per):int(len(c1) * val_per)]))
    c_val = np.vstack((c_val, c2[int(len(c2) * train_per):int(len(c2) * val_per)]))
    c_val = np.vstack((c_val, c3[int(len(c3) * train_per):int(len(c3) * val_per)]))
    c_val = np.vstack((c_val, c4[-2].reshape(1, window)))
    c_val = np.vstack((c_val, c5[-2].reshape(1, window)))
    c_val = c_val[1:]
    a_test = np.vstack((a_test, a1[int(len(a1) * val_per):]))
    a_test = np.vstack((a_test, a2[int(len(a2) * val_per):]))
    a_test = np.vstack((a_test, a3[int(len(a3) * val_per):]))
    a_test = np.vstack((a_test, a4[-1].reshape(1, window)))
    a_test = np.vstack((a_test, a5[-1].reshape(1, window)))
    a_test = a_test[1:]
    b_test = np.vstack((b_test, b1[int(len(b1) * val_per):]))
    b_test = np.vstack((b_test, b2[int(len(b2) * val_per):]))
    b_test = np.vstack((b_test, b3[int(len(b3) * val_per):]))
    b_test = np.vstack((b_test, b4[-1].reshape(1, window)))
    b_test = np.vstack((b_test, b5[-1].reshape(1, window)))
    b_test = b_test[1:]
    c_test = np.vstack((c_test, c1[int(len(c1) * val_per):]))
    c_test = np.vstack((c_test, c2[int(len(c2) * val_per):]))
    c_test = np.vstack((c_test, c3[int(len(c3) * val_per):]))
    c_test = np.vstack((c_test, c4[-1].reshape(1, window)))
    c_test = np.vstack((c_test, c5[-1].reshape(1, window)))
    c_test = c_test[1:]

    return x_train, y_train, x_val, y_val, x_test, y_test, a_train, b_train, c_train, a_val, b_val, c_val, a_test, b_test, c_test


# 根据难度划分数据集
def data_devide_10(x, y, a, b, c, window, train_per, val_per):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x0 = np.zeros((1, window, 109, 109, 3)), np.zeros(
        (1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros(
        (1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros(
        (1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3)), np.zeros((1, window, 109, 109, 3))
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y0 = [], [], [], [], [], [], [], [], [], []
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a0 = np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros(
        (1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros(
        (1, window)), np.zeros((1, window))
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b0 = np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros(
        (1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros(
        (1, window)), np.zeros((1, window))
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c0 = np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros(
        (1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros((1, window)), np.zeros(
        (1, window)), np.zeros(
        (1, window)), np.zeros((1, window))
    for i in range(len(x)):
        x_dev = x[i].reshape((1, window, 109, 109, 3))
        y_dev = y[i].tolist()
        a_dev = a[i].reshape((1, window))
        b_dev = b[i].reshape((1, window))
        c_dev = c[i].reshape((1, window))
        if 0 <= y[i] < 0.1:
            x1 = np.vstack((x1, x_dev))
            y1.append(y_dev)
            a1 = np.vstack((a1, a_dev))
            b1 = np.vstack((b1, b_dev))
            c1 = np.vstack((c1, c_dev))
        elif 0.1 <= y[i] < 0.2:
            x2 = np.vstack((x2, x_dev))
            y2.append(y_dev)
            a2 = np.vstack((a2, a_dev))
            b2 = np.vstack((b2, b_dev))
            c2 = np.vstack((c2, c_dev))
        elif 0.2 <= y[i] < 0.3:
            x3 = np.vstack((x3, x_dev))
            y3.append(y_dev)
            a3 = np.vstack((a3, a_dev))
            b3 = np.vstack((b3, b_dev))
            c3 = np.vstack((c3, c_dev))
        elif 0.3 <= y[i] < 0.4:
            x4 = np.vstack((x4, x_dev))
            y4.append(y_dev)
            a4 = np.vstack((a4, a_dev))
            b4 = np.vstack((b4, b_dev))
            c4 = np.vstack((c4, c_dev))
        elif 0.4 <= y[i] < 0.5:
            x5 = np.vstack((x5, x_dev))
            y5.append(y_dev)
            a5 = np.vstack((a5, a_dev))
            b5 = np.vstack((b5, b_dev))
            c5 = np.vstack((c5, c_dev))
        elif 0.5 <= y[i] < 0.6:
            x6 = np.vstack((x6, x_dev))
            y6.append(y_dev)
            a6 = np.vstack((a6, a_dev))
            b6 = np.vstack((b6, b_dev))
            c6 = np.vstack((c6, c_dev))
        elif 0.6 <= y[i] < 0.7:
            x7 = np.vstack((x7, x_dev))
            y7.append(y_dev)
            a7 = np.vstack((a7, a_dev))
            b7 = np.vstack((b7, b_dev))
            c7 = np.vstack((c7, c_dev))
        elif 0.7 <= y[i] < 0.8:
            x8 = np.vstack((x8, x_dev))
            y8.append(y_dev)
            a8 = np.vstack((a8, a_dev))
            b8 = np.vstack((b8, b_dev))
            c8 = np.vstack((c8, c_dev))
        elif 0.8 <= y[i] < 0.9:
            x9 = np.vstack((x9, x_dev))
            y9.append(y_dev)
            a9 = np.vstack((a9, a_dev))
            b9 = np.vstack((b9, b_dev))
            c9 = np.vstack((c9, c_dev))
        else:
            x0 = np.vstack((x0, x_dev))
            y0.append(y_dev)
            a0 = np.vstack((a0, a_dev))
            b0 = np.vstack((b0, b_dev))
            c0 = np.vstack((c0, c_dev))

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x0 = x1[1:], x2[1:], x3[1:], x4[1:], x5[1:], x6[1:], x7[1:], x8[1:], x9[1:], \
                                             x0[1:]
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a0 = a1[1:], a2[1:], a3[1:], a4[1:], a5[1:], a6[1:], a7[1:], a8[1:], a9[1:], \
                                             a0[1:]
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b0 = b1[1:], b2[1:], b3[1:], b4[1:], b5[1:], b6[1:], b7[1:], b8[1:], b9[1:], \
                                             b0[1:]
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c0 = c1[1:], c2[1:], c3[1:], c4[1:], c5[1:], c6[1:], c7[1:], c8[1:], c9[1:], \
                                             c0[1:]

    # 拼成训练集
    x_train = np.zeros((1, window, 109, 109, 3))
    y_train = np.zeros((1, 1))
    x_val = np.zeros((1, window, 109, 109, 3))
    y_val = np.zeros((1, 1))
    x_test = np.zeros((1, window, 109, 109, 3))
    y_test = np.zeros((1, 1))

    x_train = np.vstack((x_train, x1[:int(len(x1) * train_per)]))
    x_train = np.vstack((x_train, x2[:int(len(x2) * train_per)]))
    x_train = np.vstack((x_train, x3[:int(len(x3) * train_per)]))
    x_train = np.vstack((x_train, x4[:int(len(x4) * train_per)]))
    x_train = np.vstack((x_train, x5[:int(len(x5) * train_per)]))
    x_train = np.vstack((x_train, x6[:int(len(x6) * train_per)]))
    x_train = np.vstack((x_train, x7[:int(len(x7))]))
    x_train = np.vstack((x_train, x8[:int(len(x8))]))
    x_train = np.vstack((x_train, x9[:int(len(x9))]))
    x_train = np.vstack((x_train, x0[:int(len(x0))]))
    x_train = x_train[1:]
    x_val = np.vstack((x_val, x1[2:3]))
    x_val = np.vstack((x_val, x2[int(len(x2) * train_per):int(len(x2) * val_per)]))
    x_val = np.vstack((x_val, x3[int(len(x3) * train_per):int(len(x3) * val_per)]))
    x_val = np.vstack((x_val, x4[int(len(x4) * train_per):int(len(x4) * val_per)]))
    x_val = np.vstack((x_val, x5[int(len(x5) * train_per):int(len(x5) * val_per)]))
    x_val = np.vstack((x_val, x6[int(len(x6) * train_per):int(len(x6) * val_per)]))
    x_val = x_val[1:]
    x_test = np.vstack((x_test, x1[3:]))
    x_test = np.vstack((x_test, x2[int(len(x2) * val_per):]))
    x_test = np.vstack((x_test, x3[int(len(x3) * val_per):]))
    x_test = np.vstack((x_test, x4[int(len(x4) * val_per):]))
    x_test = np.vstack((x_test, x5[int(len(x5) * val_per):]))
    x_test = np.vstack((x_test, x6[int(len(x6) * val_per):]))
    x_test = x_test[1:]

    y_train = np.vstack((y_train, np.array(y1[:int(len(y1) * train_per)])))
    y_train = np.vstack((y_train, np.array(y2[:int(len(y2) * train_per)])))
    y_train = np.vstack((y_train, np.array(y3[:int(len(y3) * train_per)])))
    y_train = np.vstack((y_train, np.array(y4[:int(len(y4) * train_per)])))
    y_train = np.vstack((y_train, np.array(y5[:int(len(y5) * train_per)])))
    y_train = np.vstack((y_train, np.array(y6[:int(len(y6) * train_per)])))
    y_train = np.vstack((y_train, np.array(y7[:int(len(y7))])))
    y_train = np.vstack((y_train, np.array(y8[:int(len(y8))])))
    y_train = np.vstack((y_train, np.array(y9[:int(len(y9))])))
    y_train = np.vstack((y_train, np.array(y0[:int(len(y0))])))
    y_train = y_train[1:]
    y_val = np.vstack((y_val, np.array(y1[2:3])))
    y_val = np.vstack((y_val, np.array(y2[int(len(y2) * train_per):int(len(y2) * val_per)])))
    y_val = np.vstack((y_val, np.array(y3[int(len(y3) * train_per):int(len(y3) * val_per)])))
    y_val = np.vstack((y_val, np.array(y4[int(len(y4) * train_per):int(len(y4) * val_per)])))
    y_val = np.vstack((y_val, np.array(y5[int(len(y5) * train_per):int(len(y5) * val_per)])))
    y_val = np.vstack((y_val, np.array(y6[int(len(y6) * train_per):int(len(y6) * val_per)])))
    y_val = y_val[1:]
    y_test = np.vstack((y_test, np.array(y1[3:])))
    y_test = np.vstack((y_test, np.array(y2[int(len(y2) * val_per):])))
    y_test = np.vstack((y_test, np.array(y3[int(len(y3) * val_per):])))
    y_test = np.vstack((y_test, np.array(y4[int(len(y4) * val_per):])))
    y_test = np.vstack((y_test, np.array(y5[int(len(y5) * val_per):])))
    y_test = np.vstack((y_test, np.array(y6[int(len(y6) * val_per):])))
    y_test = y_test[1:]

    # 附加参数
    a_train = np.zeros((1, window))
    b_train = np.zeros((1, window))
    c_train = np.zeros((1, window))
    a_val = np.zeros((1, window))
    b_val = np.zeros((1, window))
    c_val = np.zeros((1, window))
    a_test = np.zeros((1, window))
    b_test = np.zeros((1, window))
    c_test = np.zeros((1, window))
    a_train = np.vstack((a_train, a1[:int(len(a1) * train_per)]))
    a_train = np.vstack((a_train, a2[:int(len(a2) * train_per)]))
    a_train = np.vstack((a_train, a3[:int(len(a3) * train_per)]))
    a_train = np.vstack((a_train, a4[:int(len(a4) * train_per)]))
    a_train = np.vstack((a_train, a5[:int(len(a5) * train_per)]))
    a_train = np.vstack((a_train, a6[:int(len(a6) * train_per)]))
    a_train = np.vstack((a_train, a7[:int(len(a7))]))
    a_train = np.vstack((a_train, a8[:int(len(a8))]))
    a_train = np.vstack((a_train, a9[:int(len(a9))]))
    a_train = np.vstack((a_train, a0[:int(len(a0))]))
    a_train = a_train[1:]
    b_train = np.vstack((b_train, b1[:int(len(b1) * train_per)]))
    b_train = np.vstack((b_train, b2[:int(len(b2) * train_per)]))
    b_train = np.vstack((b_train, b3[:int(len(b3) * train_per)]))
    b_train = np.vstack((b_train, b4[:int(len(b4) * train_per)]))
    b_train = np.vstack((b_train, b5[:int(len(b5) * train_per)]))
    b_train = np.vstack((b_train, b6[:int(len(b6) * train_per)]))
    b_train = np.vstack((b_train, b7[:int(len(b7))]))
    b_train = np.vstack((b_train, b8[:int(len(b8))]))
    b_train = np.vstack((b_train, b9[:int(len(b9))]))
    b_train = np.vstack((b_train, b0[:int(len(b0))]))
    b_train = b_train[1:]
    c_train = np.vstack((c_train, c1[:int(len(c1) * train_per)]))
    c_train = np.vstack((c_train, c2[:int(len(c2) * train_per)]))
    c_train = np.vstack((c_train, c3[:int(len(c3) * train_per)]))
    c_train = np.vstack((c_train, c4[:int(len(c4) * train_per)]))
    c_train = np.vstack((c_train, c5[:int(len(c5) * train_per)]))
    c_train = np.vstack((c_train, c6[:int(len(c6) * train_per)]))
    c_train = np.vstack((c_train, c7[:int(len(c7))]))
    c_train = np.vstack((c_train, c8[:int(len(c8))]))
    c_train = np.vstack((c_train, c9[:int(len(c9))]))
    c_train = np.vstack((c_train, c0[:int(len(c0))]))
    c_train = c_train[1:]
    a_val = np.vstack((a_val, a1[2:3]))
    a_val = np.vstack((a_val, a2[int(len(a2) * train_per):int(len(a2) * val_per)]))
    a_val = np.vstack((a_val, a3[int(len(a3) * train_per):int(len(a3) * val_per)]))
    a_val = np.vstack((a_val, a4[int(len(a4) * train_per):int(len(a4) * val_per)]))
    a_val = np.vstack((a_val, a5[int(len(a5) * train_per):int(len(a5) * val_per)]))
    a_val = np.vstack((a_val, a6[int(len(a6) * train_per):int(len(a6) * val_per)]))
    a_val = a_val[1:]
    b_val = np.vstack((b_val, b1[2:3]))
    b_val = np.vstack((b_val, b2[int(len(b2) * train_per):int(len(b2) * val_per)]))
    b_val = np.vstack((b_val, b3[int(len(b3) * train_per):int(len(b3) * val_per)]))
    b_val = np.vstack((b_val, b4[int(len(b4) * train_per):int(len(b4) * val_per)]))
    b_val = np.vstack((b_val, b5[int(len(b5) * train_per):int(len(b5) * val_per)]))
    b_val = np.vstack((b_val, b6[int(len(b6) * train_per):int(len(b6) * val_per)]))
    b_val = b_val[1:]
    c_val = np.vstack((c_val, c1[2:3]))
    c_val = np.vstack((c_val, c2[int(len(c2) * train_per):int(len(c2) * val_per)]))
    c_val = np.vstack((c_val, c3[int(len(c3) * train_per):int(len(c3) * val_per)]))
    c_val = np.vstack((c_val, c4[int(len(c4) * train_per):int(len(c4) * val_per)]))
    c_val = np.vstack((c_val, c5[int(len(c5) * train_per):int(len(c5) * val_per)]))
    c_val = np.vstack((c_val, c6[int(len(c6) * train_per):int(len(c6) * val_per)]))
    c_val = c_val[1:]
    a_test = np.vstack((a_test, a1[3:]))
    a_test = np.vstack((a_test, a2[int(len(a2) * val_per):]))
    a_test = np.vstack((a_test, a3[int(len(a3) * val_per):]))
    a_test = np.vstack((a_test, a4[int(len(a4) * val_per):]))
    a_test = np.vstack((a_test, a5[int(len(a5) * val_per):]))
    a_test = np.vstack((a_test, a6[int(len(a6) * val_per):]))
    a_test = a_test[1:]
    b_test = np.vstack((b_test, b1[3:]))
    b_test = np.vstack((b_test, b2[int(len(b2) * val_per):]))
    b_test = np.vstack((b_test, b3[int(len(b3) * val_per):]))
    b_test = np.vstack((b_test, b4[int(len(b4) * val_per):]))
    b_test = np.vstack((b_test, b5[int(len(b5) * val_per):]))
    b_test = np.vstack((b_test, b6[int(len(b6) * val_per):]))
    b_test = b_test[1:]
    c_test = np.vstack((c_test, c1[3:]))
    c_test = np.vstack((c_test, c2[int(len(c2) * val_per):]))
    c_test = np.vstack((c_test, c3[int(len(c3) * val_per):]))
    c_test = np.vstack((c_test, c4[int(len(c4) * val_per):]))
    c_test = np.vstack((c_test, c5[int(len(c5) * val_per):]))
    c_test = np.vstack((c_test, c6[int(len(c6) * val_per):]))
    c_test = c_test[1:]

    return x_train, y_train, x_val, y_val, x_test, y_test, a_train, b_train, c_train, a_val, b_val, c_val, a_test, b_test, c_test


#  5维VIT数据并储存为npy
def getVIT5dimensions_shape_image(window):
    files = os.listdir('./Data_VIT100')
    VIT5_ALL = np.zeros((1, window, 109, 109, 3))
    for f in files:
        path = os.path.join('./Data_VIT100/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        VIT5 = np.zeros((int(len(bat_dict)), window, 109, 109, 3))
        # 取 b1cj块电池的VIT数据
        for j in range(len(bat_dict)):
            # 取一块电池前100个循环的VIT三维数据
            for i in range(window):
                # 读取
                V = np.array([bat_dict['b1c' + str(j)][str(i)]['V']]).reshape(-1, 1)
                I = np.array([bat_dict['b1c' + str(j)][str(i)]['I']]).reshape(-1, 1)
                T = np.array([bat_dict['b1c' + str(j)][str(i)]['T']]).reshape(-1, 1)
                V = V.reshape(1, -1)
                I = I.reshape(1, -1)
                T = T.reshape(1, -1)

                # 转为RP矩阵并且归一化
                V_RP = RP(V)
                I_RP = RP(I)
                T_RP = RP(T)
                # 转为VIT三维矩阵
                VIT3 = np.zeros((109, 109, 3))
                VIT3[:, :, 0] = V_RP
                VIT3[:, :, 1] = I_RP
                VIT3[:, :, 2] = T_RP
                # VIT五维矩阵
                VIT5[j, i, :, :, :] = VIT3
            print('working' + str(j))
        VIT5_ALL = np.vstack((VIT5_ALL, VIT5))
    VIT5_ALL = VIT5_ALL[1:, :, :, :, :]
    np.save("./DataNpy/VIT5dimensions_image_" + str(window) + ".npy", VIT5_ALL)
    print(1)


#  获取每个cycle的IR(可以考虑其他的cycles参数，例如拐点)
def getCycles_IR(window):
    files = os.listdir('./Data3')
    IRCT_ALL = np.zeros((1, window))
    for f in files:
        path = os.path.join('./Data3/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        IRCT = np.zeros((len(bat_dict), window))
        for i in range(len(bat_dict)):
            if f == '2017-05-12.pkl':
                IRCT[i, :] = bat_dict['b1c' + str(i)]['summary']['IR'][1:window + 1]
            else:
                IRCT[i, :] = bat_dict['b1c' + str(i)]['summary']['IR'][:window]
            IRCT[i, :] = filter_limit(IRCT[i, :], 0.01)
        IRCT_ALL = np.vstack((IRCT_ALL, IRCT))
    IRCT_ALL = IRCT_ALL[1:, :]
    np.save("./DataNpy/IR.npy", IRCT_ALL)
    print(1)


#  获取每个cycle的CT(可以考虑其他的cycles参数，例如拐点)
def getCycles_CT(window):
    files = os.listdir('./Data3')
    IRCT_ALL = np.zeros((1, window))
    for f in files:
        path = os.path.join('./Data3/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        IRCT = np.zeros((len(bat_dict), window))
        for i in range(len(bat_dict)):
            if f == '2017-05-12.pkl':
                IRCT[i, :] = bat_dict['b1c' + str(i)]['summary']['chargetime'][1:window + 1]
            else:
                IRCT[i, :] = bat_dict['b1c' + str(i)]['summary']['chargetime'][:window]
            IRCT[i, :] = filter_limit(IRCT[i, :], 10)
        IRCT_ALL = np.vstack((IRCT_ALL, IRCT))
    IRCT_ALL = IRCT_ALL[1:, :]
    np.save("./DataNpy/CT.npy", IRCT_ALL)
    print(1)


#  获取Y_train数据并储存为npy
def getY_train():
    files = os.listdir('./Data3')
    QD3_ALL = np.zeros((1, 1))
    for f in files:
        path = os.path.join('./Data3/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        QD3 = np.zeros((len(bat_dict), 1))
        for i in range(len(bat_dict)):
            k = find_nearest(bat_dict['b1c' + str(i)]['summary']['QD'], 0.9)
            QD3[i, 0] = k
        QD3_ALL = np.vstack((QD3_ALL, QD3))
    QD3_ALL = QD3_ALL[1:, :]
    np.save("./DataNpy/QD3dimensions_image.npy", QD3_ALL)
    print(1)


def get_IC(window):
    files = os.listdir('./Data3')
    IC_ALL = np.zeros((1, window, 100))
    for f in files:
        path = os.path.join('./Data3/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        IC_bat = np.zeros((len(bat_dict), window, 100))
        for j in range(len(bat_dict)):
            IC_c = np.zeros((window, 100))
            if path == './Data3/2017-05-12.pkl':
                for i in range(window):  # len(bat_dict['b1c' + str(j)]['cycles'])
                    IC_c[i] = -bat_dict['b1c' + str(j)]['cycles'][str(i + 1)]['dQdV'][0:1000:10]
            else:
                for i in range(window):
                    IC_c[i] = -bat_dict['b1c' + str(j)]['cycles'][str(i)]['dQdV'][0:1000:10]
            IC_bat[j] = IC_c
        IC_ALL = np.vstack((IC_ALL, IC_bat))
    IC_ALL = IC_ALL[1:]
    np.save("./DataNpy/IC.npy", IC_ALL)
    print(1)


def get_QV50(window):
    files = os.listdir('./Data3')
    Q50_ALL = np.zeros((1, window, 1000))
    for f in files:
        path = os.path.join('./Data3/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        QV50_cycle = np.zeros((len(bat_dict), window, 1000))
        for j in range(len(bat_dict)):
            if f == '2017-05-12.pkl':
                for i in range(window - 1):
                    QV50_cycle[j, i] = bat_dict['b1c' + str(j)]['cycles'][str(i + 1)]['Qdlin'] - \
                                       bat_dict['b1c' + str(j)]['cycles'][str(1)]['Qdlin']
            else:
                for i in range(window):
                    QV50_cycle[j, i] = bat_dict['b1c' + str(j)]['cycles'][str(i)]['Qdlin'] - \
                                       bat_dict['b1c' + str(j)]['cycles'][str(0)]['Qdlin']
        Q50_ALL = np.vstack((Q50_ALL, QV50_cycle))
    Q50_ALL = Q50_ALL[1:]
    np.save("./DataNpy/QV50.npy", Q50_ALL)
    print(1)


def get_WSD_IC(window):
    IC = np.load('./DataNpy/IC.npy')
    IC_scaled = (IC - IC.min()) / (IC.max() - IC.min())
    WSD_IC = np.zeros((len(IC_scaled), window))
    for i in range(len(IC_scaled)):
        t = []
        for j in range(len(IC_scaled[i])):
            P = IC_scaled[i, 0, :]
            Q = IC_scaled[i, j, :]
            D = scipy.stats.wasserstein_distance(P, Q)
            t.append(D)
        WSD_IC[i] = np.array(t)
    np.save("./DataNpy/WSD_IC.npy", WSD_IC)
    print(1)


def get_WSD_QV50(window):
    QV50 = np.load('./DataNpy/QV50.npy')
    QV50_scaled = (QV50 - QV50.min()) / (QV50.max() - QV50.min())
    WSD_QV50 = np.zeros((len(QV50_scaled), window))
    for i in range(len(QV50_scaled)):
        t = []
        for j in range(len(QV50_scaled[i])):
            P = QV50_scaled[i, 0, :]
            Q = QV50_scaled[i, j, :]
            D = scipy.stats.wasserstein_distance(P, Q)
            t.append(D)
        WSD_QV50[i] = np.array(t)
    np.save("./DataNpy/WSD_QV50.npy", WSD_QV50)
    print(1)


def get_PICC(window):
    IC = np.load('./DataNpy/IC.npy')
    IC_scaled = (IC - IC.min()) / (IC.max() - IC.min())
    PICC = np.zeros((len(IC_scaled), window))
    for i in range(len(IC_scaled)):
        t = []
        for j in range(len(IC_scaled[i])):
            D = IC_scaled[i, j, :].max()
            t.append(D)
        # plt.plot(t)
        # plt.show()
        PICC[i] = np.array(t)
    np.save("./DataNpy/PICC.npy", PICC)
    print(1)


def get_Qd(window):
    files = os.listdir('./Data3')
    QD_ALL = np.zeros((1, window))
    for f in files:
        path = os.path.join('./Data3/' + f)
        print(path)
        bat_dict = pickle.load(open(path, 'rb'))
        QD = np.zeros((len(bat_dict), window))
        for i in range(len(bat_dict)):
            if f == '2017-05-12.pkl':
                QD[i, :] = bat_dict['b1c' + str(i)]['summary']['QD'][1:window + 1]
            else:
                QD[i, :] = bat_dict['b1c' + str(i)]['summary']['QD'][:window]
            QD[i, :] = filter_limit(QD[i, :], 0.3)
        QD_ALL = np.vstack((QD_ALL, QD))
    QD_ALL = QD_ALL[1:, :]
    np.save("./DataNpy/QD.npy", QD_ALL)
    print(1)
