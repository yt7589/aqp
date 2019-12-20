#
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from app.pytorch.diabetes_model import DiabetesModel
from app.pytorch.diabetes_dataset import DiabetesDataset

class DiabetesApp(object):
    def __init__(self):
        self.name = ''
        
    def load_dataset(self):
        '''
        train_set = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST',
            train=True,
            download=True,
            transform=([
                tvtf.ToTensor()
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=10, shuffle=True
        )
        test_set = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST',
            train=False,
            download=True,
            transform=([
                tvtf.ToTensor()
            ])
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=10, shuffle=False
        )
        '''
        dataset = DiabetesDataset()
        return DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    def run(self):
        train_loader = self.load_dataset()
        model = DiabetesModel()
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(300):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                y_hat = model(inputs)
                loss = criterion(y_hat, labels)
                print('{0}: {1}'.format(epoch, loss.data.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            '''
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            print('{0}: {1}'.format(epoch, loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
        print(model.parameters)

        x1 = Variable(torch.tensor([[6,148,72,35,0,33.6,0.627,50]], dtype=torch.float32))
        y1_hat = model.forward(x1).data[0][0]
        print('y={0}'.format(y1_hat))