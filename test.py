if __name__ == '__main__':

    import numpy as np
    import os
    import pandas as pd

    path = r'C:\Users\王小明\Desktop'
    fn = r'00371sleep 60sec every epoch.csv'
    fd = os.path.join(path, fn)

    # data = np.loadtxt(open(fd, 'rb'), delimiter=',' , skiprows=4,usecols=[6, 13])
    datas = pd.read_csv(fd, sep=',', header=3 , usecols=[6, 13])
    # datas 里面有step和sleep or awake 两个东西

    steps = datas.values[:, 0]
    # 取datas中，step里所有的数据
    steps = datas.values[1440:2880, 0]
    # 取datas中，step里8月30号一天的数据
    sum = steps.sum()
    # 对它求和，得10362
    sums = []
    for i in range(11, 15):
        # 取12天到15天的数据
        steps = datas.values[i*1440:1440*(i+1), 0]
        # 求和
        sum = steps.sum()
        sums.append(sum)

    sums = []
    for i in range(22, 30):
        # 取12天到15天的数据，早七到晚七的数据
        steps = datas.values[(i*720 + 420):(720*(i+1)+420), 0]
        # 求和
        sum = steps.sum()
        sums.append(sum)

    _ = []
    for data in datas.values[:,1]:
        if data =='S':
            flag = 0
        else:
            flag = 1
        _.append(flag)
    SoA = np.array(_)

    SoAs = []
    for i in range(22, 30):
        # 取12天到15天的数据，早七到晚七的数据醒来的数据
        soa = SoA[(i * 720 + 420):(720 * (i + 1) + 420)]
        # 求和
        sum = soa.sum()
        SoAs.append(sum)

