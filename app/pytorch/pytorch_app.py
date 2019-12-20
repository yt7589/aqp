#
from app.pytorch.fmnist import FMnist
from app.pytorch.linear_regression_app import LinearRegressionApp
from app.pytorch.logistic_regression_app import LogisticRegressionApp
from app.pytorch.diabetes_app import DiabetesApp
from app.pytorch.mnist_app import MnistApp
from app.pytorch.mnist_cnn_app import MnistCnnApp
# book
from app.pytorch.book.chp001.chp001_c001 import Chp001C001
from app.pytorch.book.chp001.chp001_c002 import Chp001C002
from app.pytorch.book.chp001.chp001_c003 import Chp001C003
from app.pytorch.book.chp001.chp001_c004 import Chp001C004
from app.pytorch.book.chp001.chp001_c005 import Chp001C005
from app.pytorch.book.chp001.chp001_c006 import Chp001C006
from app.pytorch.book.chp001.chp001_c007 import Chp001C007
from app.pytorch.book.chp001.chp001_c008 import Chp001C008
from app.pytorch.book.chp001.chp001_c009 import Chp001C009

class PyTorchApp(object):
    def __init__(self):
        self.name = 'app.pytorch.PyTorchApp'

    def startup(self):
        print('CNN之MNIST应用')
        #f_mnist = FMnist()
        #f_mnist.train()
        # *****************************
        #app = LinearRegressionApp()
        #app = LogisticRegressionApp()
        #app = DiabetesApp()
        #app = MnistApp()
        #app = MnistCnnApp()
        app = Chp001C009()
        app.run()