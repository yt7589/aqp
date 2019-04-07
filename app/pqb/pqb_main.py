from app.pqb.chp022 import Chp022
from app.pqb.chp023 import Chp023
from app.pqb.aqt001 import Aqt001

def startup():
    print('量化投资以python为工具')
    aqt001 = Aqt001()
    aqt001.startup()
    
    
    '''
    chp023 = Chp023()
    chp023.startup()
    
    chp022 = Chp022()
    chp022.startup()
    '''