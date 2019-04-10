from app.pqb.chp022 import Chp022
from app.pqb.chp023 import Chp023
from app.pqb.aqt001 import Aqt001
from app.pqb.aqt002 import Aqt002

def startup():
    print('量化投资以python为工具')
    aqt002 = Aqt002()
    aqt002.startup()
    
    
    '''
    chp023 = Chp023()
    chp023.startup()
    
    chp022 = Chp022()
    chp022.startup()
    
    aqt001 = Aqt001()
    aqt001.startup()
    '''