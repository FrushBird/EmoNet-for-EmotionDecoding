# 把这俩文件移到软件包里就会报错，很奇怪
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

class Sample:
    def __init__(self):
        self.spikes_units = []
        self.units = 0
        self.condi = None
        self.start = None
        self.end = None
        self.stimulon = None
        self.stimuloff = None

    def __len__(self):
        return len(self.spikes_units)

    def __getitem__(self, index):
        return self.spikes_units[index]

    def set_value(self,event):
        self.start = event[0]
        self.end = event[1]
        self.condi = event[3]-30
        self.stimulon = event[5]
        self.stimuloff = event[6]

    def add_spikes_unit(self,spikes):
        spikes_unit = deepcopy(spikes)
        ans1 = spikes_unit.ts > self.stimulon
        ans2 = spikes_unit.ts < self.stimulon + 1.5
        spikes_unit.ts = deepcopy(spikes_unit.ts[ans1&ans2])
        spikes_unit.ts = spikes_unit.ts- self.stimulon
        self.spikes_units.append(spikes_unit)
        self.units = self.units + 1

    def add_spikes_units(self, spikes_file):
        for spikes_unit in spikes_file:
            self.add_spikes_unit(spikes_unit)


    def __call__(self, event, spikes_file):
        self.set_value(event)
        return self.add_spikes_units(spikes_file)

    def draw_PSAS(self):

        #post stimulation all spikes_units

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('PSTH Plot')
        plt.xlabel('time')
        plt.ylabel('sample')

        for i in range(len(self.spikes_units)):
            y = np.zeros(len(self.spikes_units[i])) +i*0.01+0.01
            ax1.scatter( self.spikes_units[i],y, c='r', marker='o')
            # 显示所画的图

        plt.show()


class FR_Sample(Sample):
    def __init__(self):
        super(FR_Sample, self).__init__()
        self.FRs_units = []

    def add_spikes_unit(self,spikes):

        spikes_unit = deepcopy(spikes)
        ans1 = spikes_unit.ts > self.stimulon
        ans2 = spikes_unit.ts < self.stimulon + 2
        spikes_unit.ts = deepcopy(spikes_unit.ts[ans1&ans2])
        spikes_unit.ts = spikes_unit.ts- self.stimulon
        # 这里把它处理过了
        self.spikes_units.append(spikes_unit)
        self.units = self.units + 1

        FRs_unit = np.array([])
        # 这个是方波的算法
        # bin = 25  #单位是ms
        # for i in range(0,2000,bin):
        #     ans1 = spikes_unit.ts > 0.001*i
        #     ans2 = spikes_unit.ts < 0.001*(i+bin)
        #     ans3 = ans1 & ans2
        #     ans4 = ans3.sum()
        #     ans4 = ans4*1000/bin
        #     FRs_unit = np.append(FRs_unit, ans4)
        FRs_unit = np.array([])
        window,step = 80, 20  # 单位是ms
        for i in range(0, 2000-window+step, step):
            ans1 = spikes_unit.ts > 0.001 * i
            ans2 = spikes_unit.ts < 0.001 * (i + window)
            ans3 = ans1 & ans2
            ans4 = ans3.sum()
            ans4 = ans4 * 1000 / window
            FRs_unit = np.append(FRs_unit, ans4)

        self.FRs_units.append(FRs_unit)




class Temporal_Sample(Sample):

    def __init__(self):
        super(Temporal_Sample, self).__init__()
        self.Temporal_units = []

    def add_spikes_unit(self,spikes):

        spikes_unit = deepcopy(spikes)
        ans1 = spikes_unit.ts > self.stimulon
        ans2 = spikes_unit.ts < self.stimulon + 2
        spikes_unit.ts = deepcopy(spikes_unit.ts[ans1&ans2])
        spikes_unit.ts = spikes_unit.ts- self.stimulon
        # 这里把它处理过了
        self.spikes_units.append(spikes_unit)
        self.units = self.units + 1

        Temporal_unit = np.array([])
        bin = 5  # 单位是mius
        for i in range(0,2048,bin):
            ans1 = spikes_unit.ts > 0.001*i
            ans2 = spikes_unit.ts < 0.001*(i+bin)
            ans3 = ans1 & ans2
            ans4 = ans3.sum()
            Temporal_unit = np.append(Temporal_unit, ans4)

        self.Temporal_units.append(Temporal_unit)

        # FR和Temporal本质是一回事，仅仅精度不同。