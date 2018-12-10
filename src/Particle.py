"""
A particle.
"""
from math import sqrt, atan2
import numpy as np
from numpy import zeros, eye, size
from Sensor import ListSensorMeasurements, SensorMeasurement
from Control import Control
import Constants as C
import copy

EPS = C.EPS

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
    n_aug = C.n_aug
    lambda_aug = C.lambda_aug
    wg_aug = C.wg_aug
    wc_aug = C.wc_aug
    n_f_a = C.n_f_a
    lambda_f_a = C.lambda_f_a
    wg_f_a = C.wg_f_a
    wc_f_a = C.wc_f_a

    def __init__(self, weight, xv = zeros((3,1)), Pv = EPS*(eye(3)), Kaiy = np.empty([3, C.NPARTICLES]), \
                xf = np.empty([2,0]), Pf = np.empty([2, 2, 0]), zf = np.empty([2, 0]), idf = np.empty([1, 0]), zn = np.empty([2, 0]) ):
        self.w  = weight        # Initial weight
        self.xv = xv            # Initial vehicle pose
        self.Pv = Pv            # Initial robot covariance that considers a numerical error
        self.Kaiy = Kaiy        # Temporal keeping for a following measurement update
        self.xf = xf            # Feature mean states
        self.Pf = Pf            # Feature covariances
        self.zf = zf            # Known feature locations
        self.idf = idf          # Known feature index
        self.zn = zn            # New feature locations


    def predictACFRu(self, ctrl):
        """
        #TO DO
        Predict the state of the car given the current velocity V and steering G.
        Modifies the particles predicted mean and covariance (xv, Pv), and sigma points (Kaiy).
        :param ctrl: object of type Control.
        """
        #print("predictACFRu")

    def data_associateNN(self, z):
        """
        #TO DO
        Implements a simple gated nearest-neighbour data-association.
        Modifies the particle's zf, idf and zn.
        :param z: list of of ListSensorMeasurements. NB: Can be empty.
        """
        if(size(z)==0):
            return
        R = Particle.Re
        G_REJ = Particle.GATE_REJECT
        G_AUG = Particle.GATE_AUGMENT
        zf, zn, idf = np.array([]), np.array([]), np.array([])
        Nf = size(self.xf, 1) # number of known features
        xv = self.xv
        zp = zeros((2, 1))

        for list in z:
            for meas in list.list: # Find the nearest feature
                jbest = -1;
                if Nf != 0 :
                    jbest_s = -1
                    outer = float("inf")

                    for j in range(Nf):  # For any known feature
                        dx = self.xf[0,j]-xv[0]
                        dy = self.xf[1,j]-xv[1]
                        d = sqrt(dx**2 + dy**2)             # Distance vehicle-feaure
                        ang = pi_to_pi(atan2(dy, dx)-xv[2])
                        v = np.array([[meas.distance - d],[pi_to_pi(meas.angle-ang)]])
                        d2 = np.dot(np.transpose(v),v)
                        if(d2 < dmin):
                            dmin = d2
                            jbest_s = j

                    # Malahanobis test for the candidate neighbour
                    nis = self.__compute_association_nis(meas, R, jbest_s) #nearest neighbor
                    if nis < G_REJ :    # if within gate, store nearest neighbor
                        jbest = jbest_s;
                    elif nis < G_AUG :  # else store best nis value
                        outer = nis

                if jbest >=0:
                    zf = np.stack(zf, [m.distance, m.angle])
            self.zf, self.idf, self.zn = zf, idf, zn

    def __compute_association_nis(self, z, R, idf):
        """
        Returns normalised innovation squared (Malahanobis distance)
        """
        zp, _, _, Sf, Sv = self.__compute_jacobians(idf, R)
        v = z-zp                            # innovation
        v[2] = pi_to_pi(v[2])
        return np.dot(np.dot(np.transpose(v), np.inverse(Sf)), v)


    def __compute_jacobians(self, idf, R):

        xv = self.xv
        xf = self.xf[:, idf]
        Pf = particle.Pf[:,:,idf]
        for i in range(len(idf)):
            dx = xf[0, i] - xv[0]
            dy = xf[1, i] - xv[1]
            d2 = sqrt(dx**2 + dy**2)
            d = sqrt(d2)
            zp[:, i]  = np.array([[d],[pi_to_pi(atan2(dy, dx)-xv[2])]]) # predicted
            Hv[:,:,i] = np.array([[-dx/d, -dy/d, 0],                # Jacobian wrt vehicle states
                                    [dy/d2, -dx/d2, -1]])
            Hf[:,:,i] = np.array([[dx/d, dy/d],
                                    [-dy/d2, dx/d2]])
            Sf[:,:,i] = np.dot(np.dot(Hv[:,:,i], self.Pv), np.transpose(Hv[:,:,i]))+ R
            Sv[:,:,i] = np.dot(np.dot(Hv[:,:,i], self.Pv), np.transpose(Hv[:,:,i]))+ R
        return (zp, Hv, Hf, Sf, Sv)

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
