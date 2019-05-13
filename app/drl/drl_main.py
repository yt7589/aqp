#
from app.drl.holt_winters import HoltWinters

class DrlMain(object):
    def __init__(self):
        self.name = 'DrlMain'
        
    def startup(self):
        holt_winters = HoltWinters()
        holt_winters.startup()
