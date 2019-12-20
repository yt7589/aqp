#
import torch
from torch.autograd import Variable
from app.pytorch.logistic_regression_model import LogisticRegressionModel

class LogisticRegressionApp(object):
    def __init__(self):
        self.name = ''
        
    def load_dataset(self):
        x = Variable(torch.tensor([
            [2.1, 0.1],
            [4.2, 0.8],
            [3.1, 0.9],
            [3.3, 0.2]
        ]))
        y = Variable(torch.tensor([[0.0], [1.0], [0.0], [1.0]]))
        return x, y

    def run(self):
        X_train, y_train = self.load_dataset()
        model = LogisticRegressionModel()
        criterion = torch.nn.BCELoss(size_average=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(1000):
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            print('{0}: {1}'.format(epoch, loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(model.parameters)

        x1 = Variable(torch.tensor([[3.31, 0.21]]))
        y1_hat = model.forward(x1).data[0][0]
        print('y={0}'.format(y1_hat))