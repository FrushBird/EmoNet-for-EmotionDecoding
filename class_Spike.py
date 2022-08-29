# 把这俩文件移到软件包里就会报错，很奇怪
from copy import copy
import numpy as np
import pandas as pds

class Spike:

    def __init__(self):
        self.ts = []
        self.depth = 0
        self.ch = 0
        self.unit = 0

    def __getitem__(self, index):
        return self.ts[index]

    def __len__(self):
        return len(self.ts)

    # def set_value(self,ts,ch):
    #     self.ts = copy(ts)
    #     #在类中仍然遵循可变变量的赋值特性，即列表等容器的赋值是使用的是引用，所以要用拷贝来做独立的复制
    #     self.ch = ch
    #
    # def __call__(self, ts, ch):
    #     return self.set_value(ts,ch)

    # def set_value2(self,ts,ch,unit):
    #     self.ts = copy(ts)
    #     self.ch = ch
    #     self.unit = unit
    #     # 在类中仍然遵循可变变量的赋值特性，即列表等容器的赋值是使用的是引用，所以要用拷贝来做独立的复制
    #
    # def __call__(self, ts, ch,unit):
    #     return self.set_value2(ts,ch,unit)

    def set_value3(self,ts,ch,unit,depth_list):
        self.ts = copy(ts)
        self.ch = ch
        self.unit = unit
        self.depth = depth_list[self.ch]
        #深度的位置换算在之前提供

    def __call__(self, ts, ch,unit,depth_list):
        return self.set_value3(ts,ch,unit,depth_list)

    def __get__(self,index):
        return self.ts[index]



if __name__ == '__main__':

    #测试一下
    spike_clusters_dir = r'D:\RepoWxms\Sorted\Outputs-190710_faceszwnv_rm33_1.pl209-Jun-2022\spike_clusters.npy'
    spike_clusters = np.load(spike_clusters_dir)
    spike_times_dir = r'D:\RepoWxms\Sorted\Outputs-190710_faceszwnv_rm33_1.pl209-Jun-2022\spike_times.npy'
    spike_times = np.load(spike_times_dir)
    spike_map = spike_times

    cluster_info_dir = r'D:\RepoWxms\Sorted\Outputs-190710_faceszwnv_rm33_1.pl209-Jun-2022\cluster_info.tsv'
    cluster_info_read = pds.read_csv(cluster_info_dir, sep='\t')  # 读取tsv要制定seq=‘\t’
    cluster_info_read = cluster_info_read.to_numpy()

    spikes = []
    for i in range(len(cluster_info_read)):
        spike_ts = []
        if cluster_info_read[i, 5] == 'good':
            for j in range(len(spike_times)):
                if spike_clusters[j] == cluster_info_read[i, 0]:
                    spike_ts.append(spike_times[j, 0])
            spike_ts = spike_ts / 40000  # 很奇怪，ks提出来的ts会乘采样率
            spike = Spike()
            spike(spike_ts,cluster_info_read[i,2])
            spikes.append(spike)

    print(len(spikes[-1].ts))