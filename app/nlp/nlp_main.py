#
import os
from app.nlp.transformer_app import TransformerApp

class NlpMain(object):
    def __init__(self):
        self.name = 'NlpMain'

    def startup(self):
        print('自然语言处理Transformer模型')
        te = TransformerApp()
        te.startup()