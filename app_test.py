#
import torch
import numpy as np
from app.pytorch.pytorch_app import PyTorchApp

def main():
    app = PyTorchApp()
    app.startup()
       
    
if '__main__' == __name__:
    main()