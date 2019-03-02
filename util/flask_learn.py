from flask import Flask

class FlaskLearn(object):
    def __init__(self):
        self.name = 'FlaskLearn'
    
    def startup(self):
        print('Flask学习程序')

    def hello_world(self):
        app = Flask('HelloWorld')
        
