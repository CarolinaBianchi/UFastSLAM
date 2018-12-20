"""
A particle.
"""
from math import tan,cos,sin,sqrt,atan2,pi,exp
import numpy as np
import numpy.matlib
from numpy import zeros, eye, size, linalg
from Sensor import ListSensorMeasurements, SensorMeasurement
from Control import Control
import Constants as C
import copy
from math import sin, cos, nan
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

    def __init__(self, weight, xv = zeros((3,1)), Pv = EPS*(eye(3)), Kaiy = np.empty([3, C.NPARTICLES]), \
                xf = np.empty([2,0]), Pf = np.empty([2, 2, 0]), zf = np.empty([2, 0]), idf = np.empty([1, 0]), zn = [] ):
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
        V = ctrl.speed
        G = ctrl.steering
        dimv = self.xv.shape[0] # pose vehicle dimension
        dimQ = len(self.Qe) # measurement dimension
        # state augmentation: process noise only
        x_aug = np.append(self.xv , zeros((dimQ, 1))) # to add control input and observation

        P_aug = np.append(
            np.append(self.Pv, zeros((dimv, dimQ)), axis = 1),
            np.append(zeros((dimQ, dimv)), self.Qe, axis = 1),
            axis = 0
        )
        # set sigma points
        Z = (self.nr +self.lambda_r) * (P_aug) + EPS * eye(self.nr) # values inside the sqaure root
        S = np.transpose(linalg.cholesky(Z))
        Kaix = zeros((self.nr, 2 * self.nr + 1)) # to include both positive and negative position particles
        Kaix[:,0] = x_aug  # the average step is one of the points chosen
        for k in range(self.nr):
            # we omit the average point as already added
            Kaix[:, k + 1] = x_aug + S[:, k] # for k= 1:L
            Kaix[:, k + 1 + self.nr] = x_aug - S[:, k] # for k= L+1:2L

        Kaiy = zeros((dimv, 2 * self.nr + 1)) # array where the transformed sigma points saved with non augmented state
        xv_p = np.zeros((1, 3)) # new average state vehicle
        Pv_p = np.zeros((3,3)) #[[0 for x in range(3)] for y in range(3)]

        for index, sigma_point in enumerate(Kaix.T):
            Vn = V + sigma_point[3] # add process noise of linear speed if exists in Kaix
            Gn = G + sigma_point[4] # add process noise of steering if exist in Kaix

            Vc = Vn / (1 - tan(Gn) * self.vehicle.H / self.vehicle.L) # tan of radians ; vehicle[1] --> H ; [0] --> L

            Kaiy[0, index] = sigma_point[0] + self.dt * (Vc * cos(sigma_point[2]) - Vc / self.vehicle.L * tan(Gn) * (
                        self.vehicle.a * sin(sigma_point[2]) + self.vehicle.b * cos(sigma_point[2])))
            Kaiy[1, index] = sigma_point[1] + self.dt * (Vc * sin(sigma_point[2]) + Vc / self.vehicle.L * tan(Gn) * (
                        self.vehicle.a * cos(sigma_point[2]) - self.vehicle.b * sin(sigma_point[2])))
            Kaiy[2, index] = sigma_point[2] + Vc * self.dt * tan(Gn) / self.vehicle.L
            xv_p = xv_p + self.wg[index] * Kaiy[:,index] # average calculated by giving certain weight each particle

        self.xv = xv_p.T
        for index, sigma_point in enumerate(Kaix.T):
            d = Kaiy[:,index] - xv_p
            Pv_p = Pv_p + self.wc[index] * (d.T).dot(d)

        self.Pv = Pv_p
        self.Kaiy = Kaiy

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
        zf, zn, idf = [],[],[]
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
                    zf.append([meas.distance, meas.angle])
                    idf.append(jbest)
                elif outer > G_AUG :
                    zn.append([meas.distance, meas.angle])


            self.zf, self.idf, self.zn = np.array(zf), np.array(idf), np.array(zn).T

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
        if size(self.zf)==0:
            return
        R = Particle.Re
        n = Particle.n_aug
        lmb = Particle.lambda_aug
        wg = Particle.wg_aug
        wc = Particle.wc_aug

        lenidf = np.size(self.idf) # number of currently observed features
        dimv = self.xv.shape[0] # vehicle state dimension
        dimf = self.zf.shape[1] # feature state dimension
        # TODO: Why self.zf provides info in row format and not in columns?
        z_hat = zeros((dimf * lenidf, 1)) # predictive observation
        z = zeros((dimf * lenidf, 1)) # sensory observation
        A = zeros((dimf * lenidf, 2 * self.n_aug + 1)) # stack of innovation covariance for vehicle uncertainty
        wc_s = np.sqrt(wc)
        A_eval = zeros(np.shape(A)) # CAROLINA HECTOR LOOK HERE IDK
        Ksi = zeros((self.n_aug, 2 * self.n_aug + 1)) # SAME
        xfi = np.empty([2, 1])
        for i in range(lenidf):
            j = self.idf[i] # index of this observed feature
            xfi[:,0] = self.xf[:,j] # get j-th feature mean
            Pfi = self.Pf[:,:,j] # get j-th feature cov.
            # TODO: Check why zf had dimension num_obs x 2 instead of 2 x num_obs
            z[2 * i : 2 * i + 2, 0] = self.zf[i,:] # stack of sensory observations

            # state augmentation
            x_aug = np.append(self.xv, xfi, axis=0)  # to add control input and observation
            P_aug = np.append(
                np.append(self.Pv, zeros((dimv, dimf)), axis=1),
                np.append(zeros((dimf, dimv)), Pfi, axis=1),
                axis=0
            )
            # set sigma points
            Ps = (self.n_aug +self.lambda_aug) * P_aug + EPS * eye(self.n_aug)
            Ss = np.transpose(linalg.cholesky(Ps))
            Ksi = zeros((self.n_aug, 2 * self.n_aug + 1))
            Ksi[:,0] = x_aug[:,0]
            for k in range(self.n_aug):
                Ksi[:, k + 1] = x_aug[:,0] + Ss[:,k]
                Ksi[:, k + 1 + self.n_aug] = x_aug[:,0] - Ss[:,k]
            # passing through observation model
            Ai = zeros((dimf, 2 * n + 1)) # dim (measurement, number particles)
            bs = zeros(2 * n + 1) # bearing sign
            z_hati = zeros((2,1)) # predicted observation('dimf' by 1)
            for k in range(2 * n + 1): # pass the sigma pts through the observation model
                d = Ksi[dimv:, k] - Ksi[:dimv-1, k] # distance between particle and feature
                r = linalg.norm(d) # range
                bearing = atan2(d[1], d[0])
                bs[k] = np.sign(bearing)
                if k > 1: # unify the sign
                    if bs[k] != bs[k-1]:
                        if bs[k] < 0 and -pi < bearing and bearing < -pi/2:
                            bearing = bearing + 2 * pi
                            bs[k] = np.sign(bearing)
                        elif bs[k] > 0 and pi/2 < bearing and bearing < pi:
                            bearing = bearing - 2 * pi
                            bs[k] = np.sign(bearing)
                # distance + angle ; bearing ** do not use pi_to_pi here **
                Ai[:,k] = np.append(np.array([r]),np.array([bearing - Ksi[dimv-1, k]]),axis = 0)
                z_hati[:,0] = z_hati[:,0] + wg[k] * Ai[:,k] # predictive observation
            z_hati_rep = np.matlib.repmat(z_hati, 1, 2 * self.n_aug + 1)
            #z_hati_rep = np.matlib.repmat(z_hati, 1,self.n_aug)
            A[2 * i: 2 * i +2,:] = Ai - z_hati_rep
            A_eval = zeros(np.shape(A))
            for k in range(2 * n + 1):
                # CHANGED WITH RESPECT MATLAB IMPLEMENTATION
                A_eval[2 * i: 2 * i + 2, k] = A[2 * i: 2 * i + 2, k] * wc_s[k]

            z_hati[1] = pi_to_pi(z_hati[1]) # now use pi_to_pi for angle with respect car of possible landmark
            z_hat[2 * i: 2 * i + 2,:] = z_hati

        # augmented noise matrix
        R_aug = zeros((dimf * lenidf, dimf * lenidf))
        for i in range(lenidf):
            R_aug[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = self.Re

        # innovation covariance (THERE IS AN ISSUE)
        S = np.dot(A_eval, np.transpose(A)) # vehicle uncertainty + map + measurement noise
        S = (S + np.transpose(S))*0.5 + R_aug  # make symmetric for better numerical stability
        # cross covariance: considering vehicle uncertainty
        X = zeros((dimv, 2 * n + 1)) # stack
        print(np.shape(self.xv), np.shape(Ksi[:3, 1] ))
        for k in range(2 * n + 1):
            ksi_aux = Ksi[:3, k].reshape((3,1))
            X[:,k] = np.squeeze(wc_s[k] * (ksi_aux - self.xv))
        U = np.dot(X , np.transpose(A)) # cross covariance matrix ('dimv' by 'dimf * lenidf')

        # Kalman gain
        K = np.matmul(U, linalg.inv(S))

        # innovation('dimf*lenidf' by 1)
        v = z - z_hat
        for i in range(lenidf):
            v[2 * i] = pi_to_pi(v[2 * i])
        # standard Kalman update
        xv = self.xv + np.dot(K,v)
        Pv = self.Pv - linalg.multi_dot([K, S, np.transpose(K)])# CHANGED WITH RESPECT MATLAB IMPLEMENTATION

        # compute weight(parallel process): ERB for SLAM problem
        Lt = S # square matrix of 'dimf*lenidf'
        den = sqrt(2 * pi * linalg.det(Lt))
        num = exp(-0.5 * linalg.multi_dot([np.transpose(v), linalg.inv(Lt), v]))# TODO: CHANGE
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

        if len(self.zf) != 0: # Sample from proposal distribution, if we have not already done so above
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
