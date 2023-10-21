import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
import numpy as np
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyts.datasets import load_gunpoint

bat_dict = pickle.load(open('./Data/batch2.pkl', 'rb'))  # 记得加上'rb'
num_cells = bat_dict['b2c0']['summary']['QD'].shape[0]

# t_array = np.ones([num_cells, 2])
# for i in range(num_cells):
#     t_array[i] = np.array([bat_dict['b2c0']['summary']['QD'][i], bat_dict['b2c0']['summary']['cycle'][i]])
# print(t_array)

t_array = np.array([bat_dict['b2c0']['summary']['QD'],  bat_dict['b2c0']['summary']['cycle']])

'''
读取时间序列的数据
怎么读取需要你自己写
X为ndarray类型数据
'''
# Recurrence plot transformation
rp = RecurrencePlot(threshold='point', percentage=20)
X_rp = rp.transform(t_array)
print(X_rp[1].shape[0])

# # Show the results for the first time series
# plt.figure(figsize=(5, 5))
# plt.imshow(X_rp[0], cmap='binary', origin='lower')
# plt.title('Recurrence Plot', fontsize=16)
# plt.tight_layout()
# plt.show()

X, _, _, _ = load_gunpoint(return_X_y=True)
rp = RecurrencePlot(dimension=3, time_delay=3)
X_new = rp.transform(X)
rp2 = RecurrencePlot(dimension=3, time_delay=10)
X_new2 = rp2.transform(X)
plt.figure()
plt.suptitle('gunpoint_index_0')
ax1 = plt.subplot(121)
plt.imshow(X_new[0])
plt.title('Recurrence plot, dimension=3, time_delay=3')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(cax=cax)

# ax1 = plt.subplot(122)
# plt.imshow(X_new2[0])
# plt.title('Recurrence plot, dimension=3, time_delay=10')
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("right", size="5%", pad=0.2)
# plt.colorbar(cax=cax)
plt.show()




