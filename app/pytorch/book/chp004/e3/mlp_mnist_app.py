#
from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch

class MlpMnistApp(object):
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    def __init__(self):
        self.name = ''

    def run(self):
        self.load_dataset()

    def load_dataset(self):
        MlpMnistApp.PATH.mkdir(parents=True, exist_ok=True)
        URL = "http://deeplearning.net/data/mnist/"
        FILENAME = "mnist.pkl.gz"

        if not (MlpMnistApp.PATH / FILENAME).exists():
                content = requests.get(URL + FILENAME).content
                (MlpMnistApp.PATH / FILENAME).open("wb").write(content)
        with gzip.open((MlpMnistApp.PATH / FILENAME).as_posix(), "rb") as f:
            ((X_train, y_train), (X_valid, y_valid), (X_test, y_test)) = pickle.load(f, encoding="latin-1")
        plt.imshow(X_train[0].reshape((28, 28)), cmap="gray")
        plt.show()
        print('x_train:{0}; y_train:{1}'.format(X_train.shape, y_train.shape))
        X_train, y_train, X_valid, y_valid, X_test, y_test = map(
            torch.tensor, (X_train, y_train, X_valid, y_valid, X_test, y_test)
        )
        print(X_train.dtype)
