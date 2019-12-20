#
from mlp_mnist_app import MlpMnistApp

class Chp004E2C001(object):
    def __init__(self):
        self.name = ''

    def startup(self):
        app = MlpMnistApp()
        app.run(run_mode=MlpMnistApp.RUN_MODE_TRAIN, continue_train=True)

if '__main__' == __name__:
    app = Chp004E2C001()
    app.startup()