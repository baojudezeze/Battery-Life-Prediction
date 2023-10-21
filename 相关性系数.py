import csv
import logging
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


class CyrusMIC(object):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    screen_handler = logging.StreamHandler(sys.stdout)
    screen_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(module)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)

    def __init__(self, x_num=[None, None], y_num=[None, None]):
        self.x_max_num = x_num[1]
        self.x_min_num = x_num[0]
        self.y_min_num = y_num[0]
        self.y_max_num = y_num[1]

        self.x = None
        self.y = None

    def cal_mut_info(self, p_matrix):
        """
        计算互信息值
        :param p_matrix: 变量X和Y的构成的概率矩阵
        :return: 互信息值
        """
        mut_info = 0
        p_matrix = np.array(p_matrix)
        for i in range(p_matrix.shape[0]):
            for j in range(p_matrix.shape[1]):
                if p_matrix[i, j] != 0:
                    mut_info += p_matrix[i, j] * np.log2(p_matrix[i, j] / (p_matrix[i, :].sum() * p_matrix[:, j].sum()))
        self.logger.info("信息系数为：{}".format(mut_info / np.log2(min(p_matrix.shape[0], p_matrix.shape[1]))))
        return mut_info / np.log2(min(p_matrix.shape[0], p_matrix.shape[1]))

    def divide_bin(self, x_num, y_num):
        """
        指定在两个变量方向上需划分的网格数，返回概率矩阵
        :param x_num:
        :param y_num:
        :return: p_matrix
        """
        p_matrix = np.zeros([x_num, y_num])
        x_bin = np.linspace(self.x.min(), self.x.max() + 1, x_num + 1)
        y_bin = np.linspace(self.y.min(), self.y.max() + 1, y_num + 1)
        for i in range(x_num):
            for j in range(y_num):
                p_matrix[i, j] = sum([1 if (
                        self.x[value] < x_bin[i + 1] and self.x[value] >= x_bin[i] and self.y[value] < y_bin[
                    j + 1] and
                        self.y[value] >= y_bin[j]) else 0 for value in range(self.x.shape[0])]) / self.x.shape[0]
        return p_matrix

    def cal_MIC(self, x, y):
        self.x = np.array(x).reshape((-1,))
        self.y = np.array(y).reshape((-1,))
        if not self.x_max_num:
            self.x_max_num = int(round(self.x.shape[0] ** 0.3, 0))
            self.y_max_num = self.x_max_num
            self.x_min_num = 2
            self.y_min_num = 2
        mics = []
        for i in range(self.x_min_num, self.x_max_num + 1):
            for j in range(self.y_min_num, self.x_max_num + 1):
                self.logger.info("划分区间数量为：[{},{}]".format(i, j))
                mics.append(self.cal_mut_info(self.divide_bin(i, j)))
        self.logger.info("最大信息系数为:{}".format(max(mics)))
        return max(mics)


def scaled(q):
    q = (q - q.min()) / (q.max() - q.min())
    return q


if __name__ == '__main__':
    window = 50
    Qd = np.load('./DataNpy/Qd.npy')

    bat_dict1 = pickle.load(open('./Data3/2017-05-12.pkl', 'rb'))
    cycle1 = []
    for p in range(len(bat_dict1)):
        cycle1.append(bat_dict1["b1c" + str(p)]['cycle_life'])
    bat_dict2 = pickle.load(open('./Data3/2017-06-30.pkl', 'rb'))
    cycle = []
    for p in range(len(bat_dict2)):
        cycle.append(bat_dict2["b1c" + str(p)]['cycle_life'])
    bat_dict3 = pickle.load(open('./Data3/2018-04-12.pkl', 'rb'))
    cycle2 = []
    for p in range(len(bat_dict3)):
        cycle2.append(bat_dict3["b1c" + str(p)]['cycle_life'])

    cycle1 = np.array(cycle1)
    cycle1 = np.squeeze(cycle1, axis=2)
    cycle1 = np.squeeze(cycle1, axis=1)
    cycle = np.array(cycle)
    cycle = np.squeeze(cycle, axis=2)
    cycle = np.squeeze(cycle, axis=1)
    cycle2 = np.array(cycle2)
    cycle2 = np.squeeze(cycle2, axis=2)
    cycle2 = np.squeeze(cycle2, axis=1)
    # cycle = np.reshape(1)
    # cycle = scaled(cycle)
    cycle1 = np.hstack((cycle1, cycle))
    cycle1 = np.hstack((cycle1, cycle2))

    cycle1[117] = 2189
    cycle1[126] = 2237
    # 其他HI
    IC = np.load('./DataNpy/IC.npy')
    QV50 = np.load('./DataNpy/QV50.npy')
    WSD_IC = np.load('./DataNpy/WSD_IC.npy')
    WSD_QV50 = np.load('./DataNpy/WSD_QV50.npy')
    PICC = np.load('./DataNpy/PICC.npy')

    IR = np.load("./DataNpy/IR.npy")
    CT = np.load("./DataNpy/CT.npy")
    IR, CT = IR[:140], CT[:140]

    # 归一化
    IR_scaled = scaled(IR[:140, :window])
    CT_scaled = scaled(CT[:140, :window])
    Qd_scaled = scaled(Qd[:140, :window])
    IC_scaled = scaled(IC[:140, :window])
    WSD_IC_scaled = scaled(WSD_IC[:140, :window])
    WSD_QV50_scaled = scaled(WSD_QV50[:140, :window])
    PICC_scaled = scaled(PICC[:140, :window])

    QV50_new = QV50[:, 100, :]

    t = np.zeros((140, 1))
    for k in range(140):
        t[k] = np.var(IR_scaled[k, :])

    cycle_new = np.zeros((140, 150))
    for i in range(140):
        cycle_new[i, :] = cycle1[i]

    t = np.squeeze(t, axis=1)

    z = np.corrcoef(t, cycle1[:140])
    plt.scatter(t, cycle1[:140], c='g')
    # plt.show()

    # with open('./112233.csv', "r", encoding='utf-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(list(t))

    with open("./112233.csv", "r") as f:
        csv_reader_lines = csv.reader(f)  # 用csv.reader读文件
        date_PyList = []
        for one_line in csv_reader_lines:
            date_PyList.append(one_line)  # 逐行将读到的文件存入python的列表
        date_ndarray = np.array(date_PyList)  # 将python列表转化为ndarray

    x1 = date_ndarray[:, 0]
    x2 = date_ndarray[:, 1]
    x1 = x1.astype(np.float64)
    x2 = x2.astype(np.float64)
    x1 = scaled(x1)
    x2 = scaled(x2)

    z_norm = np.zeros((6, 6))

    for i in range(len(QV50[:140])):
        juzhen = np.zeros((6, window))
        juzhen[0] = Qd_scaled[i]
        juzhen[1] = WSD_IC_scaled[i]
        juzhen[2] = WSD_QV50_scaled[i]
        juzhen[3] = PICC_scaled[i]
        juzhen[4] = IR_scaled[i]
        juzhen[5] = CT_scaled[i]
        z = np.corrcoef(juzhen)
        z_norm = z_norm + z

        # 画相关性图
        # HI = ['QD', 'WSD_IC', 'WSD_Q50', 'PICC', 'IR', 'CT']
        # plt.yticks(np.arange(len(HI)), labels=HI)
        # plt.xticks(np.arange(len(HI)), labels=HI)
        # plt.imshow(z, cmap='seismic')
        # plt.tight_layout()
        # plt.colorbar()
        # plt.savefig('./picture/Pearsion相关性/b1c' + str(i) + '.jpg')
        # plt.close()
    z_norm = z_norm / len(QV50[:140])

    print(1)

    # MIC 不好使
    # Qd_S = scaled(Qd)
    # WSD_IC_S = scaled(WSD_IC)
    # x = Qd_S
    # y = WSD_IC_S
    # # x = np.arange(0, 6, 0.02)
    # # y = np.sin(x)
    # plt.scatter(x, y, c='g')
    # mic_tool = CyrusMIC()
    # mic_tool.cal_MIC(x, y)
    # plt.show()
