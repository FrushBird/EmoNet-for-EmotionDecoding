import os
import re
from copy import deepcopy
import pickle
from class_Sample import FR_Sample
import torch
from Util.func_MakeReOrginSamples import MakeReOrginFRSamples
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToPILImage


class Dataset_FR2Pic_AEtrain(torch.utils.data.Dataset):

    def __init__(self, num_data):
        dir_list = [r'D:\RepoWxms\Mydata\FR2Pic\Train',
                    r'D:\RepoWxms\Mydata\FR2Pic\Train',
                    r'D:\RepoWxms\Mydata\FR2Pic\Train',
                    r'D:\RepoWxms\Mydata\FR2Pic\Train'
                    ]

        self.dataset_dir = dir_list[num_data]

        dir_list2 = [r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics'
                    ]
        self.label_dir = dir_list2[num_data]

        fns = os.listdir(self.dataset_dir)
        samples = []
        num_labels = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+",fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            num_labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

        self.labels = []
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(256, 256)),
                                            ]
                                            )
        self.labels = []

        for i, num_label in enumerate(num_labels):
            dir = os.listdir(self.label_dir)
            path = os.path.join(self.label_dir, dir[num_label])
            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            self.labels.append(img)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_FR2Pic_AEtest(torch.utils.data.Dataset):

    def __init__(self, num_data):
        dir_list = [r'D:\RepoWxms\Mydata\FR2Pic\Test',
                    r'D:\RepoWxms\Mydata\FR2Pic\Test',
                    r'D:\RepoWxms\Mydata\FR2Pic\Test',
                    r'D:\RepoWxms\Mydata\FR2Pic\Test'
                    ]

        self.dataset_dir = dir_list[num_data]

        dir_list2 = [r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics'
                    ]
        self.label_dir = dir_list2[num_data]

        fns = os.listdir(self.dataset_dir)
        samples = []
        num_labels = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+",fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            num_labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

        self.labels = []
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(256, 256)),
                                            ]
                                            )
        self.labels = []

        for i, num_label in enumerate(num_labels):
            dir = os.listdir(self.label_dir)
            path = os.path.join(self.label_dir, dir[num_label])
            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            self.labels.append(img)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_FR2Pic_AERaw(torch.utils.data.Dataset):

    def __init__(self, num_data):
        dir_list = [r'D:\RepoWxms\Mydata\FR2Pic\Raw',
                    r'D:\RepoWxms\Mydata\FR2Pic\Raw',
                    r'D:\RepoWxms\Mydata\FR2Pic\Raw',
                    r'D:\RepoWxms\Mydata\FR2Pic\Raw'
                    ]

        self.dataset_dir = dir_list[num_data]

        dir_list2 = [r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics',
                    r'D:\RepoWxms\Mydata\FR2Pic\Pics'
                    ]
        self.label_dir = dir_list2[num_data]
        fns = os.listdir(self.dataset_dir)
        samples = []
        num_labels = []
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            ans = int(re.findall(r"\d+",fn.split('+')[1])[0])-1
            with open(path, 'rb+') as f:
                sample = FR_Sample()
                sample = pickle.loads(f.read())
            samples.append(sample)
            num_labels.append(ans)

        new_samples = MakeReOrginFRSamples(samples)
        # 返回unitXFR样本
        self.samples = deepcopy(new_samples)

        self.labels = []
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(256, 256)),
                                            ]
                                            )
        self.labels = []

        for i, num_label in enumerate(num_labels):
            dir = os.listdir(self.label_dir)
            path = os.path.join(self.label_dir, dir[num_label])
            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            self.labels.append(img)

    def __getitem__(self,index):
        return self.samples[index],self.labels[index]

    def __len__(self):
        return len(self.labels)




if __name__ == '__main__':
    dataset = Dataset_FR2Pic_AEtrain(1)
    # 可以把Tensor转成Image，方便可视化

    show = ToPILImage()

    for i,(tensor, label) in enumerate(dataset):
        print(i, tensor.shape, label.shape)
        show(label).show()