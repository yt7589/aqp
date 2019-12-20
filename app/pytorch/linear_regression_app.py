#
import torch
from torch.autograd import Variable
from app.pytorch.linear_regression_model import LinearRegressionModel

class LinearRegressionApp(object):
    def __init__(self):
        self.name = ''
        
    def load_dataset(self):
        x = Variable(torch.tensor([[1.0], [2.0], [3.0]]))
        y = Variable(torch.tensor([[2.0], [4.0], [6.0]]))
        return x, y

    def run(self):
        X_train, y_train = self.load_dataset()
        model = LinearRegressionModel()
        criterion = torch.nn.MSELoss(size_average=False)
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

        x1 = Variable(torch.tensor([[4.0]]))
        y1_hat = model.forward(x1).data[0][0]
        print('y={0}'.format(y1_hat))


    
