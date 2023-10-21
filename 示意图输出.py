import matplotlib.pyplot as plt
import pickle
from pylab import xticks, yticks, np
import PIL
import basic

# bat_dict = pickle.load(open('./Data/VIT2017-06-30test.pkl', 'rb'))  # 记得加上'rb'
# bat_dict1 = pickle.load(open('./Data/VIT2017-06-30.pkl', 'rb'))
bat_dict2 = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))

# a = bat_dict['b1c1']['99']['V']
# a = a.reshape(1, -1)
# b = basic.RP(a)
i = 0
# j = 308
# for j in range(len(bat_dict['b1c' + str(i)])):
# plt.plot(
#         # bat_dict2['b1c' + str(i)]['cycles'][str(j)]['Qd'], bat_dict2['b1c' + str(i)]['cycles'][str(j)]['V'],  # 'g-',
#         # bat_dict1['b1c' + str(i)][str(j)]['Qc'], bat_dict1['b1c' + str(i)][str(j)]['I'], 'b-',
#         # bat_dict['b1c' + str(i)][str(j)]['Qc'], bat_dict['b1c' + str(i)][str(j)]['V'], 'r-',
#     t, bat_dict2['b1c' + str(i)]['cycles']['0']['dQdV']
# )
#     plt.savefig('picture/cyclesQD/循环' + str(j) + '.jpg')
#     plt.close()
# plt.plot(
#     bat_dict['b1c' + str(i)]['0']['Qc'], bat_dict['b1c' + str(i)]['0']['I'],  # 'g-',
#     # bat_dict['b1c' + str(i)]['0']['Qc'], bat_dict1['b1c' + str(i)]['0']['I'], 'b-',
#     bat_dict['b1c' + str(i)]['308']['Qc'], bat_dict['b1c' + str(i)]['308']['I'], 'r-',
# )
# plt.show()
for i in range(len(bat_dict2['b1c0']['cycles'])):
    # plt.plot([-1, 1000], [0.8, 0.8], c='black', lw=1, ls='--')  # 临界点直线
    a = []
    # b = []
    num = bat_dict2['b1c40']['cycles']['100']['Qdlin'] - bat_dict2['b1c40']['cycles']['10']['Qdlin']
    q = bat_dict2['b1c0']['cycles'][str(i)]['dQdV']
    # for j in range(len(bat_dict2['b1c0']['cycles'][str(i)]['Tdlin'])):
    #     num = bat_dict2['b1c0']['cycles']['100']['Qdlin'] - bat_dict2['b1c0']['cycles']['10']['Qdlin']
    #     num2 = bat_dict2['b1c0']['cycles'][str(i)]['V'][j + 100] - bat_dict2['b1c0']['cycles'][str(i)]['V'][j]
    # a.append(num)
    #     b.append(num2)
    # a = np.array(a)
    # b = np.array(b)
    # dQdV = a / b
    plt.plot(bat_dict2['b1c0']['cycles'][str(i)]['Qdlin'], q)
    # t = np.arange(0, len(bat_dict2['b1c0']['cycles'][str(i)]['V']))
    # plt.plot(bat_dict2['b1c0']['cycles'][str(i)]['Qc'], bat_dict2['b1c0']['cycles'][str(i)]['V'])  # t
    # plt.xlim(0, 1400)
    #     plt.ylim(0.8, 1.1)  # QD
    #     plt.ylim(0, 0.03)  # IR
    # plt.ylim(-10, 0)  # dqdv
    # plt.ylim(25, 35)  # Tdlin
    plt.ylim(-10, 0)  # Vdlin
    # plt.ylim(25, 35)
    # plt.savefig('picture/test4/cycle' + str(i) + '.jpg')
    plt.show()
    plt.close()
print(1)
# vit = np.load('./DataNpy/VIT5demonsions.npy')
# vit2 = vit[:, :, :, 0, 0]
# vit2 = vit2 * 255
# imgV = PIL.Image.fromarray(vit2)
# imgV.show()


# 模板
# plt.plot(bat_dict['b1c0']['0']['Qc'], bat_dict['b1c0']['0']['V'])
# plt.plot(bat_dict['b2c0']['summary']['cycle'], bat_dict['b2c0']['summary']['Tavg'],
#          'b-', bat_dict['b2c1']['summary']['cycle'], bat_dict['b2c1']['summary']['Tavg'],
#          '*-', bat_dict['b2c43']['summary']['cycle'], bat_dict['b2c43']['summary']['Tavg']
#          )
