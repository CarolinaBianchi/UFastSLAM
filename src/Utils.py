import numpy as np
"""
Useful general purpose functions.
"""
from Constants import PI as pi
def pi_to_pi(angle):
    while angle > pi :
        angle = angle - 2*pi
    while angle < -pi :
        angle = angle + 2*pi
    return angle
