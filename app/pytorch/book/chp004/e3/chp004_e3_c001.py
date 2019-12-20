#
from mlp_mnist_app import MlpMnistApp

class Chp004E3C001(object):
    def __init__(self):
        self.name = ''

    def startup(self):
        app = MlpMnistApp()
        app.run()

if '__main__' == __name__:
    app = Chp004E3C001()
    app.startup()