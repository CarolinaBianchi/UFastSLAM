import matplotlib.pyplot as plt
import numpy as np

import math
from math import cos, sin, tan, pi, atan2, sqrt

N_PARTICLES = 1000
LANDMARK_SIZE = 50

class Particle:
    def __init__(self):
        self.w = 1 / N_PARTICLES # initial particle weight
        self.pose_vehicle = np.zeros(3, 1) # initial vehicle pose
        self.cov_vehicle = np.identity(3) # initial robot covariance that considers a numerical error
        self.Kaiy = [] # temporal keeping for a following measurement update
        self.mean_state_feature = [] # feature mean states
        self.cov_feature = [] # feature covariances
        self.known_loc_feat = [] # known feature locations
        self.known_indx_feat = [] # known feature index
        self.new_loc_feat = [] # New feature locations

def uslam():
