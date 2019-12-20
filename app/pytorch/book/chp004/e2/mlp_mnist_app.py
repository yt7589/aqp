#
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mlp_mnist_model import MlpMnistModel

class MlpMnistApp(object):
    RUN_MODE_TRAIN = 1
    RUN_MODE_PREDICT = 2

    def __init__(self):
        self.name = ''
        self.model_file = './data/mlp.pt'
        self.optz_file = './data/mlp_opt.pt'

    def run(self, run_mode=RUN_MODE_TRAIN, continue_train=False):
        train_loader, validate_loader, test_loader = self.load_dataset()
        model = MlpMnistModel()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if MlpMnistApp.RUN_MODE_TRAIN == run_mode:
            if continue_train:
                model.load_state_dict(torch.load(self.model_file))
                optimizer.load_state_dict(torch.load(self.optz_file))
            self.train(train_loader, validate_loader, test_loader, model, criterion, optimizer)
        else:
            print('载入模型参数并进行预测')
            model.load_state_dict(torch.load(self.model_file))
            model.eval()
            for batch_idx, (data, target) in enumerate(test_loader):
                Xt = data
                yt = target
            idx = 8
            Xt0 = Xt[idx]
            Xt0 = torch.unsqueeze(Xt0, 0)
            yt0 = yt[idx]
            rsts = self.predict(model, Xt0)
            rst = rsts[0]
            print('正确值:{0}; 预测值:{1}'.format(yt0, rst))
            # 绘制该样本
            img = Xt0[0][0].numpy()
            plt.imshow(img, cmap='gray')
            plt.show()

    def predict(self, model, X):
        y_hat = model(X)
        return torch.argmax(y_hat, dim=1)

    def train(self, train_loader, validate_loader, test_loader, model, criterion, optimizer):
        best_accuracy = 0.0
        threshold = 0.001 # 精度提高0.01算是有显著提高
        unimproved_epochs = 0
        unimproved_patience = 3
        epochs = 5 #
        for epoch in range(epochs):
            self.train_batch(epoch, model, criterion, optimizer, train_loader, validate_loader)
            validate_accuracy = self.evaluate(model, criterion, optimizer, validate_loader)
            print('验证精度：{0}; 最佳精度：{1}; 累积次数：{2}'.format(validate_accuracy, best_accuracy, unimproved_epochs))
            if validate_accuracy > best_accuracy*(1+threshold):
                best_accuracy = validate_accuracy
                unimproved_epochs = 0
                torch.save(model.state_dict(), self.model_file) # 保存模型
                torch.save(optimizer.state_dict(), self.optz_file)
            else:
                unimproved_epochs += 1
            if unimproved_epochs > unimproved_patience:
                print('Early Stopping已经运行')
                break
        test_accuracy = self.evaluate(model, criterion, optimizer, test_loader)
        print('最终精度：{0}'.format(test_accuracy))
        torch.save(model.state_dict(), self.model_file) # 保存模型
        torch.save(optimizer.state_dict(), self.optz_file)
    
    def train_batch(self, epoch, model, criterion, optimizer, train_loader, validate_loader):
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

    def test_load_dataset(self):
        rand_num = 15
        idx = 3
        train_loader, validate_loader, test_loader = self.load_dataset()
        for batch_idx, (data, target) in enumerate(test_loader):
            rand_num +=1
            Xt = data
            yt = target
            if rand_num > 3:
                break
        Xt0 = Xt[idx]
        Xt0 = torch.unsqueeze(Xt0, 0)
        # 绘制该样本
        img = Xt0[0][0].numpy()
        plt.imshow(img, cmap='gray')
        plt.show()