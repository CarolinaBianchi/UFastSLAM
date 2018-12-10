import numpy as np
"""
Useful general purpose functions.
"""
from Constants import PI as pi
def pi_to_pi(angles):
    for i in range(len(angles)):
        angle = angles[i]
        while angle > pi :
            angle = angle - 2*pi
        while angle < -pi :
            angle = angle + 2*pi
        angles[i] = angle
    return angles



#print(pi_to_pi(np.array([[3.0],[4.0]])))
