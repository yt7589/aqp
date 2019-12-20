#
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as tvtf

class FMnist(object):
    def __init__(self):
        self.name = 'app.pytorch.FMnist'

    def train(self):
        self._load_dataset()

    def _load_dataset(self):
        train_set = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=([
                tvtf.ToTensor()
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=10
        )
        #self.get_summary(train_set)
        self.get_one_sample(train_loader)
        print('v0.0.1')

    def _build_model(self):
        pass

    def get_one_sample(self, train_set):
        for i1 in iter(train_set):
            print(i1)
        print('train_set:{0}'.format(type(train_set)))
        itr1 = iter(train_set)
        print('itr1:{0}'.format(type(itr1)))
        a1 = [1, 2, 3]
        a2 = next(iter(a1))
        print(a2)
        #image, label = next(iter(train_set))
        #print('sizeof image:{0}'.format(image.shape))

    def get_summary(self, train_set):
        print('训练数据集大小：{0}; type:{1}'.format(len(train_set), type(train_set)))
        print('标签：{0}'.format(train_set.targets))
        print('类别平衡：{0}'.format(train_set.targets.bincount()))



        '''
        

        data_dir = './data/'
        tranform = tvtf.Compose([tvtf.ToTensor()])

        train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, transform=tranform, download=True)
        val_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=tranform, download=True)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=4, num_workers=4, shuffle=False)

        # 随机显示一个batch
        sample = next(iter(train_dataloader))
        print('sample:{0}; {1}'.format(type(sample), sample))
        '''