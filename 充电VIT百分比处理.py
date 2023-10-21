import pickle
import numpy as np

def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], idx


bat_dict = pickle.load(open('./Data/2017-05-12.pkl', 'rb'))  # 记得加上'rb'
VIT_dict = {}

for i in range(len(bat_dict)):  # 45块电池 len(bat_dict)
    cell_dict = {}
    for j in range(1, len(bat_dict['b1c' + str(i)]['cycles'])):  # 1块电池运行了400循环
        print("working")
        Qc = np.zeros(110)
        V = np.zeros(110)
        I = np.zeros(110)
        T= np.zeros(110)
        for num in range(110):
            Qc[num], k = find_nearest(bat_dict['b1c' + str(i)]['cycles'][str(j)]['Qc'], num * 0.01)
            V[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['V'][k]
            I[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['I'][k]
            T[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['T'][k]
            # 寻找最近的Qc
            # for k in range(len(bat_dict['b1c' + str(i)]['cycles'][str(j)]['Qc'])):  # 每个循环的1200个采样点
                # if round(bat_dict['b1c' + str(i)]['cycles'][str(j)]['Qc'][k], 2) == num * 0.01:
                #     Qc[num] = bat_dict['b1c' + str(i)]['cycles'][str(j)]['Qc'][k]
                #     break

        c_dict = {'Qc': Qc, 'V': V, 'I': I, 'T': T}
        cell_dict[str(j-1)] = c_dict

    key = 'b1c' + str(i)
    VIT_dict[key] = cell_dict

print("ok")
with open('./Data_VIT100/VIT2017-05-12test.pkl', 'wb') as fp:
    pickle.dump(VIT_dict, fp)
