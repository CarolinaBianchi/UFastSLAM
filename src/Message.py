import numpy as np
"""
Message exchanged between the main process and the plotter process.
"""
class Message():
    def __init__(self, xv, Pv, xf, Pf, z,  time):
        self.xv = xv
        self.Pv = Pv
        self.xf = xf
        self.Pf = Pf
        self.z = z
        self.time = time