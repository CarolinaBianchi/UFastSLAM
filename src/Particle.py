"""
A particle.
"""
from numpy import zeros, eye
from Sensor import ListSensorMeasurements, SensorMeasurement
from Odometry import Odometry
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
        #print("predictACFRu")

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
