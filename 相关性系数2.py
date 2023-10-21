import os
import pickle
import csv
import numpy as np
import scipy.stats
import basic


name = 'b1c1'
window = 130

bat_dict2 = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))
PICC = []
x1 = []
for i in range(window):
    PICC.append(np.min(bat_dict2[name]['cycles'][str(i)]['dQdV'][100:500]))
    for j in range(100, 500):
        if (bat_dict2[name]['cycles'][str(i)]['dQdV'][j] == np.min(
                bat_dict2[name]['cycles'][str(i)]['dQdV'][100:500])):
            x1.append(j)
PICC = np.array(PICC)
x1 = np.array(x1)
SOH = bat_dict2[name]['summary']['QD'][:window]
pc = np.corrcoef(PICC, SOH)

with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(x1)


a = []
c = []
k = 0
for i in range(0, len(bat_dict2[name]['cycles'])):
    a.append(-bat_dict2[name]['cycles'][str(i)]['dQdV'][:900])
    c.append((-bat_dict2[name]['cycles'][str(i)]['dQdV'][:900]).max())
c = np.array(c)
a = np.array(a)
for j in range(10, len(a), 10):
    b = a[j]
t = []
for i in range(len(a) - 1):
    P = a[i]
    Q = -bat_dict2[name]['cycles'][str(len(bat_dict2[name]['cycles']) - 1)]['dQdV'][:900]
    D = scipy.stats.wasserstein_distance(P, Q)
    t.append(D)
t = np.array(t)
WSDIC = basic.mean_filter(t[:window], 5)
pc = np.corrcoef(WSDIC, SOH)

path = os.path.join('./Data/2017-06-30.pkl')
bat_dict = pickle.load(open(path, 'rb'))
QV50_cycle = np.zeros((1, 1000))
for i in range(window):
    a = bat_dict[name]['cycles'][str(i)]['Qdlin']
    b = bat_dict[name]['cycles'][str(0)]['Qdlin']
    q = a - b
    QV50_cycle = np.vstack((QV50_cycle, q))
IC_scaled = (QV50_cycle - QV50_cycle.min()) / (QV50_cycle.max() - QV50_cycle.min())
WSD_IC = np.zeros(window)
t = []
for j in range(len(IC_scaled)):
    P = IC_scaled[0, :]
    Q = IC_scaled[j, :]
    D = scipy.stats.wasserstein_distance(P, Q)
    t.append(D)
WSD_QV = np.array(t)
WSD_QV = basic.mean_filter(WSD_QV[:window], 5)
pc = np.corrcoef(WSD_QV, SOH)

PQVC = []
x2 = []
for i in range(window):
    PQVC.append(np.min(QV50_cycle[i, :]))
    for j in range(1000):
        if np.min(QV50_cycle[i, :]) == QV50_cycle[i, j]:
            x2.append(j)
            break
PQVC = np.array(PQVC)
x2 = np.array(x2)
pc = np.corrcoef(PQVC, SOH)

# pc_all = np.corrcoef(PICC, PQVC, WSDIC, WSD_QV, SOH)


# 温度相关

# var = []
# for j in range(0, 500):
#     avg = []
#
#     for i in range(0, 950):
#         # plt.plot(bat_dict2['b1c11']['cycles'][str(i)]['Qdlin'], bat_dict2['b1c11']['cycles'][str(i)]['Tdlin'])
#         avg.append(
#             (bat_dict2['b1c11']['cycles'][str(j)]['Tdlin'][i + 50] - bat_dict2['b1c11']['cycles'][str(j)]['Tdlin'][
#                 i]) / 50)
#     avg = np.array(avg)
#     avg = np.var(avg)
#     var.append(avg)
# var = np.array(var)
# SOH = bat_dict2['b1c11']['summary']['QD'][:500]

# var = []
# for i in range(300):
#     var.append(np.var(bat_dict2['b1c0']['cycles'][str(i)]['Tdlin']))
# var = np.array(var)
#


# Apc = np.corrcoef(var, SOH)

juzhen = np.zeros((6, window))
juzhen[0] = PICC
juzhen[1] = x1
juzhen[2] = PQVC
juzhen[4] = WSD_QV
juzhen[3] = WSDIC
juzhen[5] = SOH

Apc_all = np.corrcoef(juzhen)

with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(Apc_all)

print(1)
