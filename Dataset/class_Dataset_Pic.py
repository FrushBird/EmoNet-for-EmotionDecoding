import torch
import os
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToPILImage

class Dataset_Pic_train(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset_dir = r'D:\RepoWxms\Mydata\Picdata\train'

        self.labels = []
        self.imgs = []

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(224, 224))]
                                            )
        fns = os.listdir(self.dataset_dir)
        label = None
        # 这循环之前声明一下，更安全
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            fn_ = fn[:-5]
            fn2 = fn[:2]
            if fn_ == 'Fearmonkey':
                label = 0
            elif fn_ == 'Happymonkey':
                label = 1
            elif fn_ == 'Neturalmonkey':
                label = 2
            elif fn2 == 'NE':
                label = 3
            elif fn2 == 'HA':
                label = 4
            elif fn2 == 'FE':
                label = 5

            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            if img.shape[0] == 3:
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


    def __len__(self):
        return len(self.labels)

class Dataset_Pic_test(torch.utils.data.Dataset):

    def __init__(self):
        self.dataset_dir = r'D:\RepoWxms\Mydata\Picdata\test'

        self.labels = []
        self.imgs = []

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(224, 224))]
                                            )
        fns = os.listdir(self.dataset_dir)
        label = None
        # 这循环之前声明一下，更安全
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            fn_ = fn[:-5]
            fn2 = fn[:2]
            if fn_ == 'Fearmonkey':
                label = 0
            elif fn_ == 'Happymonkey':
                label = 1
            elif fn_ == 'Neturalmonkey':
                label = 2
            elif fn2 == 'NE':
                label = 3
            elif fn2 == 'HA':
                label = 4
            elif fn2 == 'FE':
                label = 5

            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            if img.shape[0] == 3:
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


    def __len__(self):
        return len(self.labels)

class Dataset_PicOnlyHuman_train(torch.utils.data.Dataset):

    def __init__(self):
        self.dataset_dir = r'D:\RepoWxms\Mydata\Picdata2\train'

        self.labels = []
        self.imgs = []

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(256, 256))]
                                            )
        fns = os.listdir(self.dataset_dir)
        label = None
        #这循环之前声明一下，更安全
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            fn_ = fn[:-5]
            fn2 = fn[:2]
            if  fn2 == 'NE':
                label = 0
            elif fn2 == 'HA':
                label = 1
            elif fn2 == 'FE':
                label = 2

            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            if img.shape[0] == 3:
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class Dataset_PicOnlyHuman_test(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset_dir = r'D:\RepoWxms\Mydata\Picdata2\test'

        self.labels = []
        self.imgs = []

        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size=(256, 256))]
                                            )
        fns = os.listdir(self.dataset_dir)
        label = None
        # 这循环之前声明一下，更安全
        for fn in fns:
            path = os.path.join(self.dataset_dir, fn)
            fn_ = fn[:-5]
            fn2 = fn[:2]
            if fn2 == 'NE':
                label = 0
            elif fn2 == 'HA':
                label = 1
            elif fn2 == 'FE':
                label = 2

            # 正则表达式、findall返回的是一个列表，int转化不了一个容器，除非用map
            img = self.transform(Image.open(path))
            if img.shape[0] == 3:
                self.imgs.append(img)
                self.labels.append(label)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = Dataset_Pic_train()
    # 可以把Tensor转成Image，方便可视化
    show = ToPILImage()

    for i,(img, label) in enumerate(dataset):
        print(i, label, img.shape)

        # show(img).show()