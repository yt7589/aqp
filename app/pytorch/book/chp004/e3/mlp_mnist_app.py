#
from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from mlp_mnist_model import MlpMnistModel
import torch.nn.functional as F
from aqp_loss import AqpLoss
from aqp_base_optim import AqpBaseOptim

class MlpMnistApp(object):
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    def __init__(self):
        self.name = ''

    def run(self):
        print('has GPU:{0}'.format(torch.cuda.is_available()))
        dev = torch.device("cuda") if torch.cuda\
                    .is_available() else torch\
                    .device("cpu")
        print('It will run on {0}!'.format(dev))
        train_loader, validate_loader, test_loader \
                    = self.load_dataset()
        model = MlpMnistModel()
        model = model.to(dev)
        learning_rate = 0.1
        epochs = 5
        criterion = F.cross_entropy
        #criterion = AqpLoss.nll
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = AqpBaseOptim(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for xb,yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model.forward(xb)
                loss = criterion(pred, yb)
                #accuracy = model.accuracy(pred, yb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                valid_loss = sum(criterion(model(xb.to(dev)), yb.to(dev)) for xb, yb in validate_loader)
            print('{0}: loss={1}'.format(epoch, valid_loss / len(validate_loader)))
        accuracy = self.evaluate(dev, model, test_loader)
        print('accuracy in test dataset: {0}'.format(accuracy))

    def evaluate(self, dev, model, test_loader):
        accuracy = sum(model.accuracy(model(xb.to(dev)), yb.to(dev)) for xb, yb in test_loader)
        return accuracy / len(test_loader)


    def load_dataset(self):
        MlpMnistApp.PATH.mkdir(parents=True, exist_ok=True)
        URL = "http://deeplearning.net/data/mnist/"
        FILENAME = "mnist.pkl.gz"

        if not (MlpMnistApp.PATH / FILENAME).exists():
                content = requests.get(URL + FILENAME).content
                (MlpMnistApp.PATH / FILENAME).open("wb")\
                            .write(content)
        with gzip.open((MlpMnistApp.PATH / FILENAME).as_posix(), "rb") as f:
            ((X_train, y_train), (X_valid, y_valid), 
                        (X_test, y_test)) = pickle.load(f, encoding="latin-1")
        X_train, y_train, X_validate, y_validate, \
                    X_test, y_test = map(
            torch.tensor, (X_train, y_train, X_valid, y_valid, 
            X_test, y_test)
        )
        batch_size = 64
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, 
                    shuffle=True)
        validate_ds = TensorDataset(X_validate, y_validate)
        validate_loader = DataLoader(validate_ds, batch_size=batch_size*2, 
                    shuffle=False)
        test_ds = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2, 
                    shuffle=False)
        return train_loader, validate_loader, test_loader

