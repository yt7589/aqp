#
import numpy as np
import torch
from torch.autograd import Variable
from linear_regression_model import LinearRegressionModel

class LinearRegressionApp(object):
    def __init__(self):
        self.name = ''
        
    def load_dataset(self):
        np.random.seed(100)
        X0 = np.random.rand(10, 2)
        X = np.array(X0, dtype=np.float32)
        w = np.array([2.0, 3.0], dtype=np.float32)
        b = 2.6
        y = np.matmul(X, w) + b
        y = y.reshape(10, 1)
        print('X:{0}; \r\ny:{1}'.format(X, y))
        return Variable(torch.from_numpy(X)), \
                    Variable(torch.from_numpy(y))

    def run(self):
        X_train, y_train = self.load_dataset()
        model = LinearRegressionModel(in_features=2, out_features=1)
        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(1000):
            y_hat = model(X_train)
            loss = criterion(y_hat, y_train)
            if epoch % 50 == 0:
                print('{0}: {1}'.format(epoch, loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        w = model.get_weights().data
        bias = model.get_biases().data
        print('w:{0}; bias:{1}'.format(w, bias))
        

        x1 = Variable(torch.tensor([[4.0, 5.0]]))
        y1_hat = model.forward(x1).data[0][0]
        print('y={0}'.format(y1_hat))


    
