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
from math import sin, cos
from Utils import pi_to_pi

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
        zf, zn, idf = np.empty((2,1)),np.empty((2,1)),np.empty((2,1))
        Nf = size(self.xf, 1) # number of known features
        xv = self.xv
        zp = zeros((2, 1))

        for list in z:
            for meas in list.list: # Find the nearest feature
                jbest = -1;
                outer = float("inf")
                if Nf != 0 :
                    dmin = float("inf")
                    jbest_s = -1

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
                    zf = np.append(zf, np.array([[meas.distance], [meas.angle]]), axis = 1)
                    idf = np.append(idf, np.array([[idf], [jbest]]), axis = 1)
                elif outer > G_AUG :
                    zn = np.append(zn, np.array([[meas.distance], [meas.angle]]), axis = 1)


            self.zf, self.idf, self.zn = np.array(zf), np.array(idf), np.array(zn)

    def __compute_association_nis(self, z, R, idf):
        """
        Returns normalised innovation squared (Malahanobis distance)
        """
        zp, _, _, Sf, Sv = self.__compute_jacobians(idf, R)
        z= np.array([z.distance, z.angle])
        v = z-zp                            # innovation
        v[1] = pi_to_pi(v[1])
        Sf =np.array(Sf, dtype='float')
        return np.dot(np.dot(np.transpose(v), np.linalg.inv(Sf)), v)


    def __compute_jacobians(self, idf, R):

        xv = self.xv
        xf = self.xf[:, idf]
        Pf = self.Pf[:,:,idf]

        #for i in range(len(idf)): #WHY WAS MATLAB CODE LIKE THAT??
        dx = xf[0] - xv[0]
        dy = xf[1] - xv[1]
        d2 = sqrt(dx**2 + dy**2)
        d = sqrt(d2)
        zp = np.array([d,pi_to_pi(atan2(dy, dx)-xv[2])]) # predicted
        if(d < 1e-15):
            #print("Something's wrong")
            #Temporary dirty, this should get fixed when the implementation
            #is completed
            Hv = np.array([[-500, -500, 0],
                            [500, -500, -1]])
            Hf = np.array([[500, 500],
                        [-500, 500]])
        else:
            Hv = np.array(  [[-dx/d, -dy/d, 0],                # Jacobian wrt vehicle states
                            [dy/d2, -dx/d2, -1]])
            Hf = np.array([[dx/d, dy/d],
                            [-dy/d2, dx/d2]])
        Sf = np.dot(np.dot(Hv, self.Pv), np.transpose(Hv))+ R
        Sv = np.dot(np.dot(Hv, self.Pv), np.transpose(Hv))+ R
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
            self.xv = self.__multivariate_gauss(self.xv, self.Pv, 1)
            self.Pv = EPS * eye(3)

        self.__add_feature()

    def __multivariate_gauss(self,x, P, n):
        """
        Random sample from multivariate Gaussian distribution.
        :param x: mean vector
        :param P: covariance
        :param n: number of samples
        """
        samples = zeros((np.size(P,0), n))
        for i in range(n):
            samples[:,i] = np.random.multivariate_normal(np.squeeze(x), P)
        return samples

    def __add_feature(self):
        """
        Add new feature
        """
        R = Particle.Re
        z = self.zn
        lenz = size(z, 1)
        xf = zeros((2, lenz))
        Pf = zeros((2, 2, lenz))
        xv = self.xv

        for i in range(lenz):
            r, b = z[0, i], z[1, i]
            s = sin(xv[2]+b)
            c = cos(xv[2]+b)
            xf[:,i] = np.squeeze(np.array([xv[0]+r*c, xv[1]+r*s]))
            Gz = [[c, -r*s],
                  [s,  r*c]]
            Pf[:,:,i] = np.dot(np.dot(Gz, R),np.transpose(Gz))

        self.xf = np.concatenate((self.xf,xf), 1)
        self.Pf = np.concatenate((self.Pf, Pf), 2)

    def deepcopy(self):
        """
        Returns a deep copy of this particle.
        #TO DO: Test it to see if any reference is maintained!
        """
        return copy.deepcopy(self)
