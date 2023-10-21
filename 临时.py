import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

bat_dict2 = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))
window = 0
# plt.plot(bat_dict2['b1c0']['cycles'][str(window)]['Qd'], bat_dict2['b1c0']['cycles'][str(window)]['T'])
# plt.plot(bat_dict2['b1c0']['cycles'][str(50)]['V'], bat_dict2['b1c0']['cycles'][str(50)]['T'])
# plt.plot(bat_dict2['b1c0']['cycles'][str(100)]['Qd'], bat_dict2['b1c0']['cycles'][str(100)]['T'])
# plt.plot(bat_dict2['b1c0']['cycles'][str(200)]['Qd'], bat_dict2['b1c0']['cycles'][str(200)]['T'])
# plt.plot(bat_dict2['b1c0']['cycles'][str(300)]['Qd'], bat_dict2['b1c0']['cycles'][str(300)]['T'])
var = []
for j in range(0,100,20):
    avg = []

    for i in range(0, 950):
        # plt.plot(bat_dict2['b1c11']['cycles'][str(i)]['Qdlin'], bat_dict2['b1c11']['cycles'][str(i)]['Tdlin'])
        avg.append(
            (bat_dict2['b1c2']['cycles'][str(j)]['Tdlin'][i + 50] - bat_dict2['b1c2']['cycles'][str(j)]['Tdlin'][
                i]) / 50)
    plt.plot(avg)
    var.append(np.var(avg))
var = np.array(var)
# plt.ylim(25, 40)
print(bat_dict2['b1c1']['cycle_life'])

# plt.plot(bat_dict2['b1c9']['cycles'][str(100)]['Qdlin'], bat_dict2['b1c9']['cycles'][str(100)]['Tdlin'])
# plt.plot(bat_dict2['b1c9']['cycles'][str(200)]['Qdlin'], bat_dict2['b1c9']['cycles'][str(200)]['Tdlin'])
# plt.plot(bat_dict2['b1c9']['cycles'][str(300)]['Qdlin'], bat_dict2['b1c9']['cycles'][str(300)]['Tdlin'])
# plt.plot(bat_dict2['b1c9']['cycles'][str(400)]['Qdlin'], bat_dict2['b1c9']['cycles'][str(400)]['Tdlin'])
# plt.plot(bat_dict2['b1c9']['cycles'][str(500)]['Qdlin'], bat_dict2['b1c9']['cycles'][str(500)]['Tdlin'])
# plt.plot(bat_dict2['b1c9']['cycles'][str(600)]['Qdlin'], bat_dict2['b1c9']['cycles'][str(600)]['Tdlin'])

plt.show()
print(1)
# PICC
# for j in range(len(bat_dict2)):
#     a = []
#     for i in range(len(bat_dict2['b1c'+str(j)]['cycles'])):
#         D = (-bat_dict2['b1c'+str(j)]['cycles'][str(i)]['dQdV'][200:400]).max()
#         a.append(D)
#     a = np.array(a)
#     plt.plot(a)
#     plt.ylim(0, 20)
#     # plt.show()
#     plt.savefig('picture/PICC-可用/b1c' + str(j) + '.jpg')
#     plt.close()
# print(1)

# Q100 - Q10
# for j in range(len(bat_dict2)):
#     a = []
#     for i in range(10, len(bat_dict2['b1c' + str(j)]['cycles']) - 11):
#         D1 = bat_dict2['b1c' + str(j)]['cycles'][str(10)]['Qdlin'] - bat_dict2['b1c' + str(j)]['cycles'][str(i + 11)][
#             'Qdlin']
#         D2 = bat_dict2['b1c' + str(j)]['cycles'][str(10)]['Qdlin'] - bat_dict2['b1c' + str(j)]['cycles'][str(10)][
#             'Qdlin']
#         d = scipy.stats.wasserstein_distance(D2, D1)
#         a.append(d)
#     a = np.array(a)
#     plt.plot(a)
#     plt.ylim(0, 0.5)
#     # plt.show()
#     plt.savefig('picture/WSD-Q100-Q10-可用/b1c' + str(j) + 'cycle' + str(i) + '.jpg')
#     plt.close()

# IC
# a = []
# c = []
# for i in range(0, len(bat_dict2['b1c2']['cycles'])):
#     a.append(-bat_dict2['b1c2']['cycles'][str(i)]['dQdV'][:900])
#     c.append((-bat_dict2['b1c2']['cycles'][str(i)]['dQdV'][:900]).max())
# c = np.array(c)
# a = np.array(a)
# for j in range(10, len(a), 10):
#     b = a[j]

# WSD-IC
# t = []
# for i in range(len(a) - 1):
#     P = a[i]
#     Q = -bat_dict2['b1c2']['cycles'][str(len(bat_dict2['b1c2']['cycles']) - 1)]['dQdV'][:900]
#     D = scipy.stats.wasserstein_distance(P, Q)
#     t.append(D)
# t = np.array(t)
# plt.plot(t)
# plt.ylim(0, 0.5)
# plt.show()
# plt.savefig('picture/WSD-IC-可用/b1c' + str(j) + '.jpg')

# WSD-Q100-Q10

# files = os.listdir('./Data3')
# IC_ALL = np.zeros((1, window, 100))
# for f in files:
#     path = os.path.join('./Data3/' + f)
#     print(path)
#     bat_dict = pickle.load(open(path, 'rb'))
#     IC_bat = np.zeros((len(bat_dict), window, 100))
#     for j in range(len(bat_dict)):
#         IC_c = np.zeros((window, 100))
#         if path == './Data3/2017-05-12.pkl':
#             for i in range(window + 1):  # len(bat_dict['b1c' + str(j)]['cycles'])
#                 IC_c[i] = bat_dict['b1c' + str(j)]['cycles'][str(window + 1)]['dQdV'][0:1000:10] - \
#                           bat_dict['b1c' + str(j)]['cycles'][str(i + 1)]['dQdV'][0:1000:10]
#         else:
#             for i in range(window):
#                 IC_c[i] = bat_dict['b1c' + str(j)]['cycles'][str(window)]['dQdV'][0:1000:10] - \
#                           bat_dict['b1c' + str(j)]['cycles'][str(i)]['dQdV'][0:1000:10]
#         IC_bat[j] = IC_c
#     IC_ALL = np.vstack((IC_ALL, IC_bat))
# IC_ALL = IC_ALL[1:]
# np.save("./DataNpy/IC.npy", IC_ALL)
# print(1)


# matFilename = './Data/2017-05-12_batchdata_updated_struct_errorcorrect.mat'
# f = h5py.File(matFilename)
# batch = f['batch']
# num_cells = batch['summary'].shape[0]
# summary_Vdlin = np.zeros((num_cells, 1000))
# for i in range(num_cells):
#     summary_Vdlin[i] = np.hstack(np.array(f[batch['Vdlin'][i, 0]]))
# print(1)

# def find_nearest(a, a0):
#     idx = np.abs(a - a0).argmin()
#     return a.flat[idx], idx
#
#
# bat_dict = pickle.load(open('./Data/2018-04-12.pkl', 'rb'))  # 记得加上'rb'
# VIT_dict = {}
# for i in range(len(bat_dict)):  # 45块电池 len(bat_dict)
#     cell_dict = {}
#     for j in range(1, len(bat_dict['b1c' + str(i)]['cycles'])):  # 1块电池运行了400循环
#         print("working")
#         Qd = np.zeros(100)
#         V = np.zeros(100)
#         I = np.zeros(100)
#         T = np.zeros(100)
#         for num in range(100):
#             Qd[num], k = find_nearest(bat_dict['b1c' + str(i)]['cycles'][str(j)]['Qd'], num * 0.01)
#             V[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['V'][k]
#             I[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['I'][k]
#             T[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['T'][k]
#
#         c_dict = {'Qd': Qd, 'V': V, 'I': I, 'T': T}
#         cell_dict[str(j - 1)] = c_dict
#
#     key = 'b1c' + str(i)
#     VIT_dict[key] = cell_dict
#
# with open('./Data_Vdlin/VIT2018-04-12test.pkl', 'wb') as fp:
#     pickle.dump(VIT_dict, fp)
# print(1)

window = 50

# files = os.listdir('./Data_Vdlin')
# Q50_ALL = np.zeros((1, window, 100))
# for f in files:
#     path = os.path.join('./Data_Vdlin/' + f)
#     print(path)
#     bat_dict = pickle.load(open(path, 'rb'))
#     QV50_cycle = np.zeros((len(bat_dict), window, 100))
#     for j in range(len(bat_dict)):
#         for i in range(window):
#             QV50_cycle[j, i] = bat_dict['b1c' + str(j)][str(i)]['V'] - bat_dict['b1c' + str(j)][str(0)]['V']
#     Q50_ALL = np.vstack((Q50_ALL, QV50_cycle))
# Q50_ALL = Q50_ALL[1:]
# np.save("./DataNpy/QV50.npy", Q50_ALL)
# print(1)

# # WSD
# IC = np.load('./DataNpy/QV50.npy')
# IC_scaled = (IC - IC.min()) / (IC.max() - IC.min())
# WSD_IC = np.zeros((len(IC_scaled), window))
# for i in range(len(IC_scaled)):
#     t = []
#     for j in range(len(IC_scaled[i])):
#         P = IC_scaled[i, 0, :]
#         Q = IC_scaled[i, j, :]
#         D = scipy.stats.wasserstein_distance(P, Q)
#         t.append(D)
#     plt.plot(t)
#     plt.show()
#     WSD_IC[i] = np.array(t)
# np.save("./DataNpy/WSD_QV50.npy", WSD_IC)
# print(1)


# PICC
# IC = np.load('./DataNpy/IC.npy')
# IC_scaled = (IC - IC.min()) / (IC.max() - IC.min())
# PICC = np.zeros((len(IC_scaled), window))
# for i in range(len(IC_scaled)):
#     t = []
#     for j in range(len(IC_scaled[i])):
#         D = IC_scaled[i, j, :].max()
#         t.append(D)
#     # plt.plot(t)
#     # plt.show()
#     PICC[i] = np.array(t)
# np.save("./DataNpy/PICC.npy", PICC)
# print(1)

# Qd
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
        # IRCT[i, :] = filter_limit(IRCT[i, :], 10)
    QD_ALL = np.vstack((QD_ALL, QD))
QD_ALL = QD_ALL[1:, :]
np.save("./DataNpy/QD_" + str(window) + ".npy", QD_ALL)
print(1)
