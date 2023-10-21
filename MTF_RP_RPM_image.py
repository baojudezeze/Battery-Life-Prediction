import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyts.datasets import load_gunpoint


def GAF(X):
    '''
    the code of GAF
    input: the time series in raw
    output: the heatmap of GASF and GADF
    '''
    # normalization to [-1,1]
    X = ((X - np.max(X)) + (X - np.min(X))) / (np.max(X) + np.min(X))

    # generate the GASF
    gasf = X.T * X - np.sqrt(1 - np.square(X)).T * np.sqrt(1 - np.square(X))
    sns.heatmap(gasf, cbar=False, square=True, cmap='GnBu', xticklabels=False, yticklabels=False)
    # plt.show()
    plt.savefig('picture/GASF_1.jpg')

    # generate the GADF
    gadf = np.sqrt(1 - np.square(X)).T * X + X.T * np.sqrt(1 - np.square(X))

    # plot the heatmap
    sns.heatmap(gadf, cbar=False, square=True, cmap='GnBu', xticklabels=False, yticklabels=False)
    # plt.show()
    plt.savefig('picture/GADF_1.jpg')

    return 0


def MTF(X):
    '''
    the code of MTF
    input: the time series in raw
    output: the heatmap of MTF
    '''
    # normalization to [0,1]
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    # the length of X
    Xlen = X.shape[1]
    # design the number of bins
    Q = 4
    # compute the temp matrix X_Q
    X_Q = np.ones([1, Xlen]) * 4
    # print(X_Q)
    temp = 0
    threshold = np.zeros([1, Q])
    for i in range(Q):
        # print((Xlen * i / Q))
        # print(np.sum(X < temp))
        while np.sum(X < temp) < (Xlen * i / Q):
            temp += 0.01
        # print(threshold.shape)
        threshold[0][i] = temp
        X_Q[np.where(X < temp)] -= 1
        # print(X_Q)
    # print(threshold)

    # generate the Markov matrix
    sum_MM = np.zeros([4, 4])

    # compute the probability of Markov
    for i in range(Xlen - 1):
        if X_Q[0][i] - X_Q[0][i + 1] == -3:
            sum_MM[0][3] = sum_MM[0][3] + 1
        elif X_Q[0][i] - X_Q[0][i + 1] == -2:
            if X_Q[0][i] == 1:
                sum_MM[0][2] = sum_MM[0][2] + 1
            elif X_Q[0][i] == 2:
                sum_MM[1][3] = sum_MM[1][3] + 1
        elif X_Q[0][i] - X_Q[0][i + 1] == -1:
            if X_Q[0][i] == 1:
                sum_MM[0][1] = sum_MM[0][1] + 1
            elif X_Q[0][i] == 2:
                sum_MM[1][2] = sum_MM[1][2] + 1
            elif X_Q[0][i] == 3:
                sum_MM[2][3] = sum_MM[2][3] + 1
        elif X_Q[0][i] - X_Q[0][i + 1] == 0:
            if X_Q[0][i] == 1:
                sum_MM[0][0] = sum_MM[0][0] + 1
            elif X_Q[0][i] == 2:
                sum_MM[1][1] = sum_MM[1][1] + 1
            elif X_Q[0][i] == 3:
                sum_MM[2][2] = sum_MM[2][2] + 1
            elif X_Q[0][i] == 4:
                sum_MM[3][3] = sum_MM[3][3] + 1
        elif X_Q[0][i] - X_Q[0][i + 1] == 1:
            if X_Q[0][i] == 2:
                sum_MM[1][0] = sum_MM[1][0] + 1
            elif X_Q[0][i] == 3:
                sum_MM[2][1] = sum_MM[2][1] + 1
            elif X_Q[0][i] == 4:
                sum_MM[3][2] = sum_MM[3][2] + 1
        elif X_Q[0][i] - X_Q[0][i + 1] == 2:
            if X_Q[0][i] == 3:
                sum_MM[2][0] = sum_MM[2][0] + 1
            elif X_Q[0][i] == 4:
                sum_MM[3][1] = sum_MM[3][1] + 1
        elif X_Q[0][i] - X_Q[0][i + 1] == 3:
            sum_MM[3][0] = sum_MM[3][0] + 1
    W = sum_MM
    W = W / np.sum(W, axis=1)
    # print(W)

    # generate the Markov Transform Field
    mtf = np.zeros([Xlen, Xlen])
    for i in range(Xlen):
        for j in range(Xlen):
            # print(X_Q[0][i])
            mtf[i][j] = W[int(X_Q[0][i]) - 1][int(X_Q[0][j]) - 1]
    mtf = (mtf - mtf.min()) / (mtf.max() - mtf.min()) * 4
    # generate the heatmap
    sns.heatmap(mtf, cbar=False, square=True, cmap='GnBu', xticklabels=False, yticklabels=False)
    # plt.show()
    plt.savefig('picture/MTF_1.jpg')
    return 0


def RP(X):
    # normalization to [0,1]
    X = (X - np.max(X)) / (np.max(X) + np.min(X))
    Xlen = 50  # X.shape[1]
    # convert to the phase space(第一元素是此时高度，第二个给元素为下一时刻的高度)
    S = np.zeros([Xlen-1,2])
    S = X[0, :50]
    # S[:, 0] = X[0, 1:50]
    # S[:, 1] = X[0, 2:51]
    print(S)

    # compute RRP matrix
    R = np.zeros([Xlen - 1, Xlen - 1])
    for i in range(Xlen - 1):
        for j in range(Xlen - 1):
            # R[i, j] = sum(pow(S[i, :] - S[j, :], 2))
            R[i, j] = (S[i]-S[j])/(i-j+0.000001)
    # normalization to [0,4] of RP
    R = (R - R.min()) / (R.max() - R.min()) * 4




    # show the heatmap(bwr,coolwarm,GnBu)
    x2 = plt.subplot(122)
    sns.heatmap(R, cbar=True, square=True, xticklabels=False, yticklabels=False, center=0)
    # plt.show()
    # plt.savefig('picture/RP_1.jpg')

    return 0


def MDF(X, n):
    # normalization to [0,1]
    X = (X - np.max(X)) / (np.max(X) + np.min(X))

    # compute the length of time series and the range of d
    T = X.shape[1]
    dMax = math.floor((T - 1) / (n - 1))

    # initial the M,IMG
    M = np.zeros([n, T - n + 1, dMax])

    # initial the dM,K
    dM = np.zeros([n - 1, T - n + 1, dMax])
    K = np.ones([n - 1, T - n + 1, dMax])
    for d in range(dMax):
        d = d + 1
        # initial the s
        s = np.zeros([T - (n - 1) * d])
        for i in range(T - (n - 1) * d):
            s[i] = i
        for ImageIndex in range(n):
            # print(s+ImageIndex*d)
            s_index = (s + ImageIndex * d).astype(np.int16)
            s = s.astype(np.int16)
            # print(X[0,s_index])
            M[ImageIndex, s, d - 1] = X[0, s_index]

            if ImageIndex >= 1:
                # motif difference
                dM[ImageIndex - 1, s, d - 1] = M[ImageIndex, s, d - 1] - M[ImageIndex - 1, s, d - 1]
                # K
                K[ImageIndex - 1, s, d - 1] = np.zeros([T - (n - 1) * d])

    IMG = np.zeros([n - 1, T - n + 1, dMax])
    for ImageIndex in range(n - 1):
        # G
        G = dM[ImageIndex]
        # the rot180 of G
        G_1 = G.reshape(1, (T - n + 1) * dMax)
        G_2 = G_1[0][::-1]
        G_rot = G_2.reshape(T - n + 1, dMax)
        IMG[ImageIndex, :, :] = G + K[ImageIndex] * G_rot
        sns.heatmap(IMG[ImageIndex, :, :].T, cbar=False, square=True, cmap='GnBu', xticklabels=False, yticklabels=False)
        # plt.show()
        # print(IMG)
        plt.savefig('picture/MDF' + '%d' % ImageIndex + '_1.jpg')
    return 0


if __name__ == "__main__":
    bat_dict = pickle.load(open('./Data/batch2.pkl', 'rb'))  # 记得加上'rb'

    for i in range(48):
        data_numpy = np.array([bat_dict['b2c' + str(i)]['summary']['QD']])
        plt.figure(1)
        plt.figure(figsize=(16, 4), dpi=80)
        ax1 = plt.subplot(121)

        plt.plot(bat_dict['b2c' + str(i)]['summary']['cycle'], bat_dict['b2c' + str(i)]['summary']['QD'])
        RP(data_numpy)
        plt.savefig('picture/RP_' + str(i) + '_' + str(bat_dict['b2c' + str(i)]['summary']['QD'].shape[0]) + '.jpg')
        plt.close()


    # GAF(data_numpy)
    # MTF(data_numpy.T)
    # n = 4
    # MDF(data_numpy.T, n)
    # x=list()
    # temp = list()
    # for t in np.arange(0,10,0.01):
    #     temp.append(t)
    #     x.append(math.sin(t))
    # print(x)
    # plt.plot(temp,x)
    # plt.show()
