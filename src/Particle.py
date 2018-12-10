"""
A particle.
"""
from math import tan,cos,sin,sqrt,atan2,pi,exp
import numpy as np
from numpy import zeros, eye, linalg
from Sensor import ListSensorMeasurements, SensorMeasurement
from Control import Control
import Constants as C
import copy

EPS = C.EPS

def pi_to_pi(angle_array):
    for angle in angle_array:
        if angle > pi:
            while angle > pi:
                angle = angle - 2 * pi


        elif angle < -pi:
            while angle < -pi:
                angle = angle + 2 * pi
    return angle

class Particle:
    #Static constant attributes
    Qe = C.Qe
    vehicle = C.VEHICLE
    dt = C.DT
    nr = C.n_r              # dimv + dimQ
    lambda_r = C.lambda_r
    wg = C.wg_r
    wc = C.wc_r
    Re = C.Re
    GATE_REJECT = C.GATE_REJECT
    GATE_AUGMENT = C.GATE_AUGMENT_NN
    n_aug = C.n_aug # dimv + dimf
    lambda_aug = C.lambda_aug
    wg_aug = C.wg_aug
    wc_aug = C.wc_aug
    n_f_a = C.n_f_a
    lambda_f_a = C.lambda_f_a
    wg_f_a = C.wg_f_a
    wc_f_a = C.wc_f_a
    V = 0 # process noise linear speed
    G = 0 #process noise of steering

    def __init__(self, weight, xv = zeros((3,1)), Pv = EPS*(eye(3)), Kaiy = [], \
                xf = [], Pf = [], zf = [], idf = [], zn = [] ):
        self.w  = weight        # Initial weight
        self.xv = xv            # Initial vehicle pose
        self.Pv = Pv            # Initial robot covariance that considers a numerical error
        self.Kaiy = Kaiy        # Temporal keeping for a following measurement update
        self.xf = xf            # Feature mean states
        self.Pf = Pf            # Feature covariances
        self.zf = zf            # Known feature locations
        self.idf = idf          # Known feature index
        self.zn = zn            # New feature locations


    def predictACFRu(self, odometry):
        """
        #TO DO
        Predict the state of the car given the current velocity V and steering G.
        Modifies the particles predicted mean and covariance (xv, Pv), and sigma points (Kaiy).
        :param odometry: object of type Odometry.
        """

        dimv = np.size(self.xv, 1) # pose vehicle dimension
        dimQ = np.size(self.Qe, 1) # measurement dimension
        # state augmentation: process noise only
        x_aug = np.append(self.xv , zeros((dimQ, 1)), axis = 0) # to add control input and observation

        P_aug = np.append(
            np.append(self.Pv, zeros(dimv, dimQ), axis = 1),
            np.append(zeros(dimQ, dimv), self.Qe, axis = 1),
            axis = 0
        )
        # set sigma points
        Z = (self.nr +self.lambda_r) * (P_aug) + EPS * eye(self.nr) # values inside the sqaure root
        S = np.transpose(linalg.cholesky(Z))
        Kaix = zeros(self.nr, 2 * self.nr + 1) # to include both positive and negative position particles
        Kaix[:,0] = x_aug; # the average step is one of the points chosen
        for k in range(self.nr):
            # we omit the average point as already added
            Kaix[:, k + 1] = x_aug + S[:, k] # for k= 1:L
            Kaix[:, k + 1 + self.nr] = x_aug - S[:, k] # for k= L+1:2L

        Kaiy = zeros(dimv, 2 * self.nr + 1) # array where the transformed sigma points saved with non augmented state
        xv_p = 0 # new average state vehicle
        Pv_p = 0 # new average covariance

        for index, sigma_point in enumerate(Kaix.T):
            Vn = self.V + sigma_point[3] # add process noise of linear speed if exists in Kaix
            Gn = self.G + sigma_point[4] # add process noise of steering if exist in Kaix

            Vc = Vn / (1 - tan(Gn) * self.vehicle[1] / self.vehicle[0]) # tan of radians ; vehicle[1] --> H ; [0] --> L

            Kaiy[0, index] = sigma_point[0] + self.dt * (Vc * cos(sigma_point[2]) - Vc / self.vehicle[0] * tan(sigma_point[2]) * (
                        self.vehicle[3] * sin(sigma_point[2]) + self.vehicle[2] * cos(sigma_point[2])))
            Kaiy[1, index] = sigma_point[1] + self.dt * (Vc * sin(sigma_point[2]) - Vc / self.vehicle[0] * tan(sigma_point[2]) * (
                        self.vehicle[3] * cos(sigma_point[2]) - self.vehicle[2] * sin(sigma_point[2])))
            Kaiy[2, index] = sigma_point[2] + Vc * self.dt * tan(Gn) / self.vehicle[0]

            xv_p = xv_p + self.wg[index] * Kaiy[:,index] # average calculated by giving certain weight each particle
            Pv_p = Pv_p + self.wc[index] * (Kaiy[:,index] - xv_p)*np.transpose(Kaiy[:,index] - xv_p)

        self.xv = xv_p
        self.Pv = Pv_p
        self.Kaiy = Kaiy

    def data_associateNN(self, z):
        """
        #TO DO
        Implements a simple gated nearest-neighbour data-association.
        Modifies the particle's zf, idf and zn.
        :param z: list of of ListSensorMeasurements. NB: Can be empty.
        """
        R = Particle.Re
        G_REJ = Particle.GATE_REJECT
        G_AUG = Particle.GATE_AUGMENT
        #print(z)

    def sample_proposaluf(self):
        """
        #TO DO
        Compute proposal distribution and then sample from it.
        """
        if len(self.zf)==0:
            return
        R = Particle.Re
        n = Particle.n_aug
        lmb = Particle.lambda_aug
        wg = Particle.wg_aug
        wc = Particle.wc_aug

        lenidf = np.size(self.idf) # number of currently observed features
        dimv = np.size(self.xv) # vehicle state dimension
        dimf = np.size(self.zf) # feature state dimension
        z_hat = zeros(dimf * lenidf, 1) # predictive observation
        z = zeros(dimf * lenidf, 1) # sensory observation
        A = zeros(dimf * lenidf, 2 * self.n_aug + 1) # stack of innovation covariance for vehicle uncertainty
        wc_s = sqrt(wc)
        for i in range(lenidf):
            j = self.idf[i] # index of this observed feature
            xfi = self.xf[:,j] # get j-th feature mean
            Pfi = self.Pf[:,:,j] # get j-th feature cov.
            z[2 * i : 2 * i + 1, 1] = self.zf[:,i] # stack of sensory observations

            # state augmentation
            x_aug = np.append(self.xv, xfi, axis=0)  # to add control input and observation
            P_aug = np.append(
                np.append(self.Pv, zeros(dimv, dimf), axis=1),
                np.append(zeros(dimf, dimv), Pfi, axis=1),
                axis=0
            )
            # set sigma points
            Ps = (self.n_aug +self.lambda_aug) * P_aug + EPS * eye(self.n_aug)
            Ss = np.transpose(linalg.cholesky(Ps))
            Ksi = zeros(self.n_aug, 2 * self.n_aug + 1)
            Ksi[:,0] = x_aug
            for k in range(self.n_aug):
                Ksi[:, k + 1] = x_aug + Ss[:,k]
                Ksi[:, k + 1 + self.n_aug] = x_aug - Ss[:,k]
            # passing through observation model
            Ai = zeros(dimf, 2 * n + 1) # dim (measurement, number particles)
            bs = zeros(2 * n + 1) # bearing sign
            z_hati = 0 # predicted observation('dimf' by 1)
            for k in range(2 * n + 1): # pass the sigma pts through the observation model
                d = Ksi[dimv:, k] - Ksi[:dimv-1, k] # distance between particle and feature
                r = linalg.norm(d) # range
                bearing = atan2(d(2), d(1))
                bs[k] = np.sign(bearing);
                if k > 1: # unify the sign
                    if bs[k] != bs[k-1]:
                        if bs[k] < 0 and -pi < bearing and bearing < -pi/2:
                            bearing = bearing + 2 * pi
                            bs[k] = np.sign(bearing)
                        elif bs(k) > 0 and pi/2 < bearing and bearing < pi:
                            bearing = bearing - 2 * pi
                            bs[k] = np.sign(bearing)
                # distance + angle ; bearing ** do not use pi_to_pi here **
                Ai[:,k] = np.append(r,bearing - Ksi[dimv-1, k],axis = 0)
                z_hati = z_hati + wg[k] * Ai[:,k] # predictive observation
            z_hati_rep = np.matlib.repmat(z_hati, 1, 2 * self.n_aug + 1)
            A[2 * i: 2 * i +1,:] = Ai - z_hati_rep
            A_eval = zeros(np.size(A))
            for k in range(2 * n + 1):
                # CHANGED WITH RESPECT MATLAB IMPLEMENTATION
                A_eval[2 * i: 2 * i + 1, k] = A[2 * i: 2 * i + 1, k] * wc_s[k]

            z_hati[1] = pi_to_pi(z_hati[1]) # now use pi_to_pi for angle with respect car of possible landmark
            z_hat[2 * i: 2 * i + 1, 1] = z_hati

        # augmented noise matrix
        R_aug = zeros(dimf * lenidf, dimf * lenidf)
        for i in range(lenidf):
            R_aug[2 * i - 1: 2 * i, 2 * i - 1: 2 * i] = self.Re

        # innovation covariance (THERE IS AN ISSUE)
        S = A_eval * np.transpose(A) # vehicle uncertainty + map + measurement noise
        S = (S + np.transpose(S))*0.5 + R_aug  # make symmetric for better numerical stability
        # cross covariance: considering vehicle uncertainty
        X = zeros(dimv, 2 * n + 1) # stack
        for k in range(2 * n + 1):
            X[:,k] = wc_s[k] * (Ksi[:3, k] - self.xv)
        U = X * np.transpose(A) # cross covariance matrix ('dimv' by 'dimf * lenidf')

        # Kalman gain
        K = np.matmul(U, linalg.inv(S))

        # innovation('dimf*lenidf' by 1)
        v = z - z_hat
        for i in range(lenidf):
            v[2 * i] = pi_to_pi(v[2 * i])
        # standard Kalman update
        xv = self.xv + K * v
        Pv = self.Pv - K * S * np.transpose(K)  # CHANGED WITH RESPECT MATLAB IMPLEMENTATION

        # compute weight(parallel process): ERB for SLAM problem
        Lt = S # square matrix of 'dimf*lenidf'
        den = sqrt(2 * pi * linalg.det(Lt))
        num = exp(-0.5 * np.transpose(v) * linalg.inv(Lt) * v) # TODO: CHANGE
        w = num / den
        self.w = self.w * w

        # sample from proposal distribution
        xvs = self.__multivariate_gauss(xv, Pv, 1)
        self.xv = xvs
        self.Pv = eye(3) * EPS # initialize covariance

    def feature_updateu(self):
        """
        Having selected a new pose from the proposal distribution, this pose is
        assumed perfect and each feature update may be computed independently
        and without pose uncertainty.
        Modifies particles xf and Pf.
        """
        if len(self.zf)==0:
            return
        R = Particle.Re
        N = Particle.n_f_a
        lmb = Particle.lambda_f_a
        wg_f_a = Particle.wg_f_a
        wc_f_a = Particle.wc_f_a

    def augment_map(self):
        if len(self.zn) == 0:
            return

        if len(self.zf ==0): # Sample from proposal distribution, if we have not already done so above
            self.xv = __multivariate_gauss(self.xv, self.Pv, 1)
            self.Pv = EPS * eye(3)

        self.__add_feature()

    def __multivariate_gauss(x, P, n):
        """
        Random sample from multivariate GAussian distribution.
        :param x: mean vector
        :param P: covariance
        :param n: number of samples
        """
        return []

    def __add_feature():
        """
        Add new feature
        """
        R = Particle.Re
        z = self.ze

    def deepcopy(self):
        """
        Returns a deep copy of this particle.
        #TO DO: Test it to see if any reference is maintained!
        """
        return copy.deepcopy(self)
