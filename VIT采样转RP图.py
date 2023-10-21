import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import pickle


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
    # normalization to [0,4] of RP
    R = (R - R.min()) / (R.max() - R.min()) * 4
    # show the heatmap(bwr,coolwarm,GnBu)
    sns.heatmap(R, cbar=True, square=True, cmap='GnBu', xticklabels=False, yticklabels=False, center=0)
    # plt.show()
    # plt.savefig('picture/RP_1.jpg')
    return 0


if __name__ == "__main__":
    bat_dict = pickle.load(open('./Data/VIT2017-06-30test.pkl', 'rb'))
    for i in range(len(bat_dict['b1c0'])):
        data_numpy = []
        data_numpy = np.array([bat_dict['b1c0'][str(i)]['T']])
        RP(data_numpy)
        plt.savefig('picture/T_RP/RP_' + str(i) + '.jpg')
        plt.close()
