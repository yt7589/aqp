#
from logistic_regression_app import LogisticRegressionApp

class Chp003E1C001(object):
    def __init__(self):
        self.name = ''

    def startup(self):
        app = LogisticRegressionApp()
        app.run()

if '__main__' == __name__:
    app = Chp003E1C001()
    app.startup()