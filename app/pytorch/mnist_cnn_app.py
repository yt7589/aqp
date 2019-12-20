#
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from app.pytorch.mnist_cnn_model import MnistCnnModel

class MnistCnnApp(object):
    def __init__(self):
        self.name = ''

    def run(self):
        train_loader, validate_loader, test_loader = self.load_dataset()
        model = MnistCnnModel()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_accuracy = 0.0
        threshold = 0.01 # 精度提高0.01算是有显著提高
        unimproved_epochs = 0
        unimproved_patience = 3
        for epoch in range(50):
            self.train(epoch, model, criterion, optimizer, train_loader, validate_loader)
            validate_accuracy = self.evaluate(model, criterion, optimizer, validate_loader)
            print('验证精度：{0}; 最佳精度：{1}; 累积次数：{2}'.format(validate_accuracy, best_accuracy, unimproved_epochs))
            if validate_accuracy > best_accuracy*(1+threshold):
                best_accuracy = validate_accuracy
                unimproved_epochs = 0
            else:
                unimproved_epochs += 1
            if unimproved_epochs > unimproved_patience:
                print('Early Stopping已经运行')
                break
        test_accuracy = self.evaluate(model, criterion, optimizer, test_loader)
        print('最终精度：{0}'.format(test_accuracy))
    
    def train(self, epoch, model, criterion, optimizer, train_loader, validate_loader):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            y_hat = model(data)
            loss = criterion(y_hat, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('train epoch:{0} [{1}/{2}   ({3:.2f})]   loss:{4:.6f}'.format(epoch, 
                    batch_idx*len(data), len(train_loader.dataset), 
                    100. * batch_idx /len(train_loader), 
                    loss.data)
                )

    def evaluate(self, model, criterion, optimizer, data_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            data, target = Variable(data), Variable(target)
            y_hat = model(data)
            test_loss += criterion(y_hat, target)
            pred = torch.max(y_hat.data, 1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(data_loader.dataset)
        print('\n测试TestSet: Average_loss:{0:.4f}, Accuracy: {1}/{2} ({3:.4f})\n'.format(test_loss, 
            correct, len(data_loader.dataset), 
            100. * correct / len(data_loader.dataset))
        )

        return 1.0 * correct / len(data_loader.dataset)

        

    def test_cross_entropy(self):
        loss = torch.nn.CrossEntropyLoss()
        Y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False) # 是0、1、2，不是one-hot
        Y_1 = Variable(torch.tensor([
            [0.1, 0.2, 0.9],
            [1.1, 0.1, 0.2],
            [0.2, 2.1, 0.1]
        ])) # 是logits，不是softmax
        Y_2 = Variable(torch.tensor([
            [0.8, 0.2, 0.3],
            [0.2, 0.3, 0.5],
            [0.2, 0.2, 0.5]
        ]))

        loss1 = loss(Y_1, Y)
        loss2 = loss(Y_2, Y)
        print('loss1={0}; loss2={1}'.format(loss1.data, loss2.data))

        
    def load_dataset(self):
        batch_size = 32
        raw_train_set = torchvision.datasets.MNIST(
            root='./data/MNIST',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ])
        )
        train_set, validate_set = torch.utils.data.random_split(raw_train_set, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        validate_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=False
        )
        test_set = torchvision.datasets.MNIST(
            root='./data/MNIST',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,),(0.3081,))
            ])
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=10, shuffle=False
        )
        return train_loader, validate_loader, test_loader