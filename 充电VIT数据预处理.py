import pickle

bat_dict = pickle.load(open('./Data/2017-05-12.pkl', 'rb'))  # 记得加上'rb'
VIT_dict = {}

for i in range(len(bat_dict)):
    cycle_dict = {}
    for j in range(len(bat_dict['b1c' + str(i)]['cycles'])):
        cell_dict = {}
        V = bat_dict['b1c' + str(i)]['cycles'][str(j)]['V']
        I = bat_dict['b1c' + str(i)]['cycles'][str(j)]['I']
        T = bat_dict['b1c' + str(i)]['cycles'][str(j)]['T']
        Qc = bat_dict['b1c' + str(i)]['cycles'][str(j)]['Qc']
        cell_dict = {'V': V, 'I': I, 'T': T, 'Qc': Qc}
        key = str(j)
        cycle_dict[key] = cell_dict
    key = 'b1c' + str(i)
    VIT_dict[key] = cycle_dict
print("ok")
with open('./Data/VIT2019-09-10.pkl', 'wb') as fp:
    pickle.dump(VIT_dict, fp)
