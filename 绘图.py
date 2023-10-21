import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.ticker import MultipleLocator

import basic


def draw_VIT_V():
    VIT100 = pickle.load(open('./Data_VIT100/VIT2017-06-30test.pkl', 'rb'))
    # VIT100 = pickle.load(open('./Data3/2017-06-30.pkl', 'rb'))
    for i in range(len(VIT100)):
        V = VIT100['b1c' + str(i)]['0']['V']
        Qc = VIT100['b1c' + str(i)]['0']['Qc']
        plt.plot(Qc, V, c='red')
        plt.title('Voltage curve', fontsize=12)
        plt.xlabel('Charge capacity (Ah)', fontsize=12)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.grid(True)
        plt.savefig('paperDraw/充电V/b1c' + str(i) + '.jpg')
        plt.close()
        # plt.show()


def draw_VIT_I():
    VIT100 = pickle.load(open('./Data_VIT100/VIT2017-06-30test.pkl', 'rb'))
    # VIT100 = pickle.load(open('./Data3/2017-06-30.pkl', 'rb'))
    for i in range(len(VIT100)):
        I = VIT100['b1c' + str(i)]['0']['I']
        Qc = VIT100['b1c' + str(i)]['0']['Qc']
        plt.plot(Qc, I, c='blue')
        plt.title('Current curve', fontsize=12)
        plt.xlabel('Charge capacity (Ah)', fontsize=12)
        plt.ylabel('Current (C rate)', fontsize=12)
        plt.grid(True)
        plt.savefig('paperDraw/充电I/b1c' + str(i) + '.jpg')
        plt.close()
        # plt.show()


def draw_VIT_T():
    VIT100 = pickle.load(open('./Data_VIT100/VIT2017-06-30test.pkl', 'rb'))
    # VIT100 = pickle.load(open('./Data3/2017-06-30.pkl', 'rb'))
    for i in range(len(VIT100)):
        T = VIT100['b1c' + str(i)]['0']['T']
        Qc = VIT100['b1c' + str(i)]['0']['Qc']
        plt.plot(Qc, T, c='green')
        plt.title('Temperature curve', fontsize=12)
        plt.xlabel('Charge capacity (Ah)', fontsize=12)
        plt.ylabel('Temperature (℃)', fontsize=12)
        plt.grid(True)
        plt.savefig('paperDraw/充电T/b1c' + str(i) + '.jpg')
        plt.close()
        # plt.show()


def draw_RP_V():
    VIT5 = np.load('./DataNpy/VIT5dimensions_image_50.npy')
    V = VIT5[46:94]
    for i in range(len(V)):
        RP_V = V[i, 49, :, :, 0]
        plt.figure(figsize=(9, 7))
        plt.imshow(RP_V, cmap='jet')
        plt.tight_layout()
        # plt.colorbar()

        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.rc('font', family='Times New Roman')
        ax = plt.gca()  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(10)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(10)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(10)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(10)  # 设置上部坐标轴的粗细

        # plt.title('RP Image for Voltage', fontsize=12)
        # plt.savefig('paperDraw/RP_V/b1c' + str(i) + '.jpg')
        # plt.close()
        plt.show()
        print(1)


def draw_RP_I():
    VIT5 = np.load('./DataNpy/VIT5dimensions_image_50.npy')
    I = VIT5[46:94]
    for i in range(len(I)):
        RP_I = I[i, 0, :, :, 1]
        plt.figure(figsize=(6, 5.5))
        plt.imshow(RP_I, cmap='jet')
        plt.tight_layout()
        # plt.colorbar()

        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.rc('font', family='Times New Roman')
        ax = plt.gca()  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(10)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(10)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(10)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(10)  # 设置上部坐标轴的粗细

        # plt.title('RP Image for Current', fontsize=12)
        # plt.savefig('paperDraw/RP_I/b1c' + str(i) + '.jpg')
        # plt.close()
        plt.show()
        print(1)


def draw_RP_T():
    VIT5 = np.load('./DataNpy/VIT5dimensions_image_50.npy')
    T = VIT5[46:94]
    for i in range(len(T)):
        RP_T = T[i, 0, :, :, 2]
        plt.figure(figsize=(6, 5.5))
        plt.imshow(RP_T, cmap='jet')
        plt.tight_layout()
        # plt.colorbar()

        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.rc('font', family='Times New Roman')
        ax = plt.gca()  # 获得坐标轴的句柄
        ax.spines['bottom'].set_linewidth(10)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(10)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(10)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(10)  # 设置上部坐标轴的粗细

        # plt.title('RP Image for Temperature', fontsize=12)
        plt.savefig('paperDraw/RP_T/b1c' + str(i) + '.jpg')
        plt.close()
        # plt.show()
        print(1)


def draw_RP_VIT():
    VIT5 = np.load('./DataNpy/VIT5dimensions_image_50.npy')

    RP_T = VIT5[50, 49, :, :, 0]
    # plt.figure(figsize=(6, 5.5))
    plt.imshow(RP_T, cmap='jet')
    plt.tight_layout()
    plt.colorbar()
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('paperDraw/绘图素材/1.jpg')
    # plt.close()
    plt.show()
    print(1)


def get_b1c4(window):
    window = 400
    bat_dict = pickle.load(open('./Data_VIT100/VIT2017-06-30test.pkl', 'rb'))
    vall = np.zeros((1, 110))
    iall = np.zeros((1, 110))
    tall = np.zeros((1, 110))

    VIT5 = np.load('./DataNpy/VIT5dimensions_image_50.npy')
    V = VIT5[1, :, :, :, 0]
    ax = plt.axes(projection='3d')
    plt.plot(V)
    plt.show()

    for i in range(0, window, 20):
        V = bat_dict['b1c4'][str(i)]['V'].reshape(1, -1)
        I = bat_dict['b1c4'][str(i)]['I'].reshape(1, -1)
        T = bat_dict['b1c4'][str(i)]['T'].reshape(1, -1)
        vall = np.vstack((vall, V))
        iall = np.vstack((iall, I))
        tall = np.vstack((tall, T))

    vall = vall[1:]
    iall = iall[1:]
    tall = tall[1:]

    # with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(list(tall))

    V = bat_dict['b1c4'][str(300)]['V'].reshape(1, -1)
    I = bat_dict['b1c4'][str(10)]['I'].reshape(1, -1)
    T = bat_dict['b1c4'][str(300)]['T'].reshape(1, -1)

    V = basic.RP(V)
    I = basic.RP(I)
    T = basic.RP(T)

    with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list(T))

    plt.figure(figsize=(9, 8))
    plt.imshow(V, cmap='jet')
    plt.tight_layout()

    # plt.figure(figsize=(6, 5.5))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.rc('font', family='Times New Roman')
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(0)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(0)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(0)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(0)  # 设置上部坐标轴的粗细

    # T = 1 - T
    # plt.imshow(T, cmap='Blues')  # cmap='jet'
    # plt.tight_layout()
    # plt.colorbar()
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('paperDraw/绘图素材/1.jpg')
    # plt.close()
    plt.show()
    print(1)


def draw_PICC():
    bat_dict2 = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))
    a = []
    c = []
    for i in range(0, len(bat_dict2['b1c2']['cycles'])):
        a.append(-bat_dict2['b1c2']['cycles'][str(i)]['dQdV'][:900])
        c.append((-bat_dict2['b1c2']['cycles'][str(i)]['dQdV'][:900]).max())
    c = np.array(c)
    a = np.array(a)
    for j in range(10, len(a), 10):
        b = a[j]

    c = basic.mean_filter(c, 5)

    with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(c))

    plt.plot(c, linestyle='-', linewidth=2)  # color='b'
    plt.title('PICC (Av/h)', fontsize=12)
    plt.xlabel('Cycle number', fontsize=12)
    plt.grid(True)
    plt.ylim(0, 10)
    plt.show()
    print(1)


def draw_WSD_IC():
    bat_dict2 = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))

    a = []
    c = []
    k = 25

    for i in range(0, len(bat_dict2['b1c' + str(k)]['cycles'])):
        a.append(-bat_dict2['b1c' + str(k)]['cycles'][str(i)]['dQdV'][:900])
        c.append((-bat_dict2['b1c' + str(k)]['cycles'][str(i)]['dQdV'][:900]).max())
    c = np.array(c)
    a = np.array(a)
    for j in range(10, len(a), 10):
        b = a[j]

    t = []
    for i in range(len(a) - 1):
        P = a[i]
        Q = -bat_dict2['b1c' + str(k)]['cycles'][str(len(bat_dict2['b1c' + str(k)]['cycles']) - 1)]['dQdV'][:900]
        D = scipy.stats.wasserstein_distance(P, Q)
        t.append(D)
    t = np.array(t)

    t = basic.mean_filter(t, 5)

    with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(t))

    plt.plot(t, linestyle='-', color='r', linewidth=2)  # color='b'
    plt.title('WSD-IC', fontsize=12)
    plt.xlabel('Cycle number', fontsize=12)
    plt.grid(True)
    plt.ylim(0, 0.4)
    plt.show()
    print(1)


def draw_WSD_QV(window):
    path = os.path.join('./Data/2017-06-30.pkl')
    bat_dict = pickle.load(open(path, 'rb'))
    QV50_cycle = np.zeros((1, 1000))
    for i in range(300):
        a = bat_dict['b1c0']['cycles'][str(i)]['Qdlin']
        b = bat_dict['b1c0']['cycles'][str(0)]['Qdlin']
        q = a - b
        QV50_cycle = np.vstack((QV50_cycle, q))

    print(1)

    IC_scaled = (QV50_cycle - QV50_cycle.min()) / (QV50_cycle.max() - QV50_cycle.min())
    WSD_IC = np.zeros(window)

    t = []
    for j in range(len(IC_scaled)):
        P = IC_scaled[0, :]
        Q = IC_scaled[j, :]
        D = scipy.stats.wasserstein_distance(P, Q)
        t.append(D)
    WSD_QV = np.array(t)

    WSD_QV = basic.mean_filter(WSD_QV, 5)

    with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(t))

    plt.plot(WSD_QV, linestyle='-', color='g', linewidth=2)  # color='b'

    plt.title('WSD-ΔQ(V)', fontsize=12)
    plt.xlabel('Cycle number', fontsize=12)
    plt.grid(True)
    plt.ylim(0, 0.4)
    plt.show()
    print(1)


def draw_SOH():
    bat_dict = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))
    Qd = bat_dict['b1c4']['summary']['QD']
    Qd = basic.filter_limit(Qd, 0.1)
    Qd = Qd[:50]
    # with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(list(Qd))

    # plt.figure(figsize=(9, 7))
    plt.plot(Qd, linestyle='-', color='orangered', linewidth=5)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.rc('font', family='Times New Roman')
    y_major_locator = MultipleLocator(0.002)  # 以每3显示
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.yaxis.set_major_locator(y_major_locator)
    ax.spines['bottom'].set_linewidth(5)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(5)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(5)  # 设置底部坐标轴的粗细
    ax.spines['top'].set_linewidth(5)  # 设置左边坐标轴的粗细
    plt.ylim(1.069, 1.076)
    # plt.xlim(0, 50)
    plt.show()
    print(1)


def draw_b1c4_放电V():
    bat_dict = pickle.load(open('./Data/2017-06-30.pkl', 'rb'))
    plt.plot(bat_dict['b1c4']['cycles']['0']['Qd'][:-180], bat_dict['b1c4']['cycles']['0']['V'][:-180],
             linestyle='-', color='r', linewidth=2)
    Qd = bat_dict['b1c4']['cycles']['0']['Qd'][:-180]
    V = bat_dict['b1c4']['cycles']['0']['V'][:-180]

    # plt.title('WSD-IC', fontsize=12)
    plt.xlabel('Discharge capacity', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    # plt.grid(True)
    plt.xlim(0, )
    plt.ylim(2, 3.6)
    plt.show()


def draw_Q100_Q10():
    path = os.path.join('./Data/2017-06-30.pkl')
    bat_dict = pickle.load(open(path, 'rb'))

    t = bat_dict['b1c3']['cycles'][str(100)]['Qdlin'] - bat_dict['b1c3']['cycles'][str(10)]['Qdlin']

    with open('./112233.csv', "w", encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(t))

    plt.plot(t)
    plt.show()

    print(1)


if __name__ == "__main__":
    window = 300  # 450
    # draw_VIT_I()
    # draw_VIT_V()
    # draw_VIT_T()
    # draw_RP_V()
    # draw_RP_I()
    # draw_RP_T()
    # draw_RP_VIT()

    # 效果最好：b1c4
    # get_b1c4(window)
    # draw_SOH()

    # draw_PICC()
    # draw_WSD_IC()
    draw_WSD_QV(window)

    # draw_b1c4_放电V()
    # draw_Q100_Q10()
    print(1)
