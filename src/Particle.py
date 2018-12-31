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
    Re = C.Re # Observation(measurement) noises covariance
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
        self.xf = xf            # Feature mean states -- in global coordinates
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
        L = self.nr                 # number of dimensions of the augmented state vector
        dt = self.dt                # time between each control input

        #Car parameters
        a, b, Le, H = self.vehicle.a, self.vehicle.b, self.vehicle.L, self.vehicle.H
        V = ctrl.speed
        G = ctrl.steering
        dimv = self.xv.shape[0]     # pose vehicle dimension
        dimQ = len(self.Qe)         # control dimension; Qe [variance speed ,0 ; 0, variance steering]

        # State augmentation: process noise only
        x_aug = np.append(self.xv , zeros((dimQ, 1))) # to add control input

        # (EQ 2)
        P_aug = np.append(
            np.append(self.Pv, zeros((dimv, dimQ)), axis = 1),
            np.append(zeros((dimQ, dimv)), self.Qe, axis = 1),
            axis = 0
        )

        # Set sigma points
        Z = (L +self.lambda_r) * (P_aug) + EPS * eye(L) # values inside the square root
        S = linalg.cholesky(Z)
        Kaix = zeros((L, 2 * L + 1))                    # to include both positive and negative position particles
        Kaix[:,0] = x_aug                               # the average step is one of the points chosen

        # (EQ 3)
        for k in range(L):
            # we omit the average point as already added
            Kaix[:, k + 1] = x_aug + S[:, k]            # for k= 1:L
            Kaix[:, k + 1 + L] = x_aug - S[:, k]  # for k= L+1:2L

        Kaiy = zeros((dimv, 2 * L + 1))                       # process model evaluated in sigma points
        xv_p = np.zeros((1, 3)) # new average state vehicle
        Pv_p = np.zeros((3,3)) #[[0 for x in range(3)] for y in range(3)]

        for index, sigma_point in enumerate(Kaix.T):
            Vn = V + sigma_point[3] # add process noise of linear speed if exists in Kaix
            Gn = G + sigma_point[4] # add process noise of steering if exist in Kaix

            Vc = Vn / (1 - tan(Gn) * H / Le)

            # (EQ 4)
            Kaiy[0, index] = sigma_point[0] + dt * (Vc * cos(sigma_point[2]) - Vc / Le * tan(Gn) * (
                        a * sin(sigma_point[2]) + b * cos(sigma_point[2])))
            Kaiy[1, index] = sigma_point[1] + dt * (Vc * sin(sigma_point[2]) + Vc / Le * tan(Gn) * (
                        a * cos(sigma_point[2]) - b * sin(sigma_point[2])))
            Kaiy[2, index] = sigma_point[2] + Vc * dt * tan(Gn) / Le

            # (EQ 5)
            xv_p = xv_p + self.wg[index] * Kaiy[:,index] # average calculated by giving certain weight each sigma point

        self.xv = xv_p.T

        # (EQ 6)
        for index, sigma_point in enumerate(Kaix.T):
            d = Kaiy[:,index] - xv_p
            Pv_p = Pv_p + self.wc[index] * (d.T).dot(d)

        self.Pv = Pv_p
        self.Kaiy = Kaiy

    def data_associateNN(self, z):
        """
        Implements a simple gated nearest-neighbour data-association.
        Modifies the particle's zf, idf and zn.
        :param z: list of of ListSensorMeasurements. NB: Can be empty.
        """
        R = Particle.Re
        G_REJ = Particle.GATE_REJECT
        G_AUG = Particle.GATE_AUGMENT
        zf, zn, idf = [],[],[]
        Nf = size(self.xf, 1) # number of known features
        xv = self.xv
        zp = zeros((2, 1))

        for meas in z:

            jbest = -1
            outer = float("inf")
            if Nf != 0 :
                dmin = float("inf")
                jbest_s = -1

                for j in range(Nf):  # For any known feature
                    dx = self.xf[0,j]-xv[0]             # distance_x between known feature and current car pose
                    dy = self.xf[1,j]-xv[1]
                    d = sqrt(dx**2 + dy**2)             # Distance vehicle-feaure
                    ang = pi_to_pi(atan2(dy, dx)-xv[2])
                    v = np.array([[meas.distance - d],[pi_to_pi(meas.angle-ang)]])
                    d2 = np.dot(np.transpose(v),v)
                    if(d2 < dmin):
                        dmin = d2
                        jbest_s = j # put it the index of the known feature more probable

                # Malahanobis test for the candidate neighbour
                nis = self.__compute_association_nis(meas, R, jbest_s) #nearest neighbor
                if nis < G_REJ :    # if within gate, store nearest neighbor
                    jbest = jbest_s
                elif nis < G_AUG :  # else store best nis value
                    outer = nis

            if jbest >= 0:
                zf.append([meas.distance, meas.angle])
                idf.append(jbest)
            elif outer > G_AUG : # if no features saved yet it will get inside here
                zn.append([meas.distance, meas.angle])


        self.zf, self.idf, self.zn = np.array(zf), np.array(idf), np.array(zn).T

    def __compute_association_nis(self, z, R, idf):
        """
        Returns normalised innovation squared (Malahanobis distance)
        """
        zp, Sf = self.__compute_jacobians(idf, R)
        z= np.array([z.distance, z.angle])
        v = z-zp                            # innovation
        v[1] = pi_to_pi(v[1])
        Sf =np.array(Sf, dtype='float')
        return (v.T.dot(np.linalg.inv(Sf))).dot(v)


    def __compute_jacobians(self, idf, R):

        xv = self.xv
        xf = self.xf[:, idf]
        Pf = self.Pf[:,:,idf]

        #for i in range(len(idf)):
        dx = xf[0] - xv[0]
        dy = xf[1] - xv[1]
        d2 = dx**2 + dy**2
        d = sqrt(d2)
        zp = np.array([d,pi_to_pi(atan2(dy, dx)-xv[2])]) # predicted measure from car frame to the feature

        #Hv = np.array(  [[-dx/d, -dy/d, 0],                # Jacobian wrt vehicle states
        #                [dy/d2, -dx/d2, -1]])
        Hf = np.array([[dx/d, dy/d],                       # Jacobian wrt feature states
                        [-dy/d2, dx/d2]])
        Sf = np.dot(np.dot(np.squeeze(Hf), Pf), np.transpose(np.squeeze(Hf))) + R
        #Sv = np.dot(np.dot(Hv, self.Pv), np.transpose(Hv))+ R
        return (zp, Sf)

    def sample_proposaluf(self):
        """
        Compute proposal distribution and then sample from it.
        """
        R = Particle.Re
        n = Particle.n_aug
        wg = Particle.wg_aug
        wc = Particle.wc_aug
        n_aug = self.n_aug
        lenidf = np.size(self.idf)                          # number of currently observed features
        dimv = self.xv.shape[0]                             # vehicle state dimension
        dimf = self.zf.shape[1]                             # feature state dimension
        # TODO: Why self.zf provides info in row format and not in columns?
        n_hat = zeros((dimf * lenidf, 1))                   # predictive observation
        z = zeros((dimf * lenidf, 1))                       # sensory observation
        N = zeros((dimf * lenidf, 2 * self.n_aug + 1))      # stack of innovation covariance for vehicle uncertainty
        N_eval = zeros(np.shape(N))
        wc_s = np.sqrt(wc)
        xfi = np.empty([2, 1])

        for i in range(lenidf):                             # each feature observed in this timestep
            j = self.idf[i]                                 # index of this observed feature
            xfi[:,0] = self.xf[:,j]                         # get j-th feature mean
            Pfi = self.Pf[:,:,j]                            # get j-th feature cov.

            # TODO: Check why zf had dimension num_obs x 2 instead of 2 x num_obs
            z[2 * i : 2 * i + 2, 0] = self.zf[i,:]          # stack of sensory observations

            # state augmentation
            x_aug = np.append(self.xv, xfi, axis=0)         # to add control input and observation that agree with known features
            P_aug = np.append(
                np.append(self.Pv, zeros((dimv, dimf)), axis=1),
                np.append(zeros((dimf, dimv)), Pfi, axis=1),
                axis=0
            )

            # set sigma points for this feature
            Ps = (n_aug + self.lambda_aug) * P_aug + EPS * eye(n_aug)
            Ss = linalg.cholesky(Ps)
            Ksi = zeros((n_aug, 2 * n_aug + 1))
            Ksi[:,0] = x_aug[:,0]
            for k in range(n_aug):
                Ksi[:, k + 1] = x_aug[:,0] + Ss[:,k]
                Ksi[:, k + 1 + n_aug] = x_aug[:,0] - Ss[:,k]

            # passing through observation model
            Ni = zeros((dimf, 2 * n + 1))           # dim (measurement, number particles)
            bs = zeros(2 * n + 1)                   # bearing sign
            n_hati = zeros((2,1))                   # predicted observation('dimf' by 1)
            # (EQ 8)
            for k in range(2 * n + 1):              # pass the sigma pts through the observation model
                d = Ksi[dimv:, k] - Ksi[:dimv-1, k] # distance between particle and feature saved before
                r = linalg.norm(d)                  # theoretical range if xv was the true state
                bearing = atan2(d[1], d[0])         # bearing
                bs[k] = np.sign(bearing)
                if k > 0: # unify the sign
                    if bs[k] != bs[k-1]:
                        if bs[k] < 0 and -pi < bearing and bearing < -pi/2:
                            bearing = bearing + 2 * pi
                            bs[k] = np.sign(bearing)
                        elif bs[k] > 0 and pi/2 < bearing and bearing < pi:
                            bearing = bearing - 2 * pi
                            bs[k] = np.sign(bearing)

                # distance + angle ; bearing ** do not use pi_to_pi here **
                Ni[:,k] = np.append(np.array([r]),np.array([bearing - Ksi[dimv-1, k]]),axis = 0)

                # (EQ 9) - Weighted mean for each sigma point
                n_hati[:,0] = n_hati[:,0] + wg[k] * Ni[:,k] # predictive observation of known feature from current pose

            n_hati_rep = np.matlib.repmat(n_hati, 1, 2 * self.n_aug + 1)
            N[2 * i: 2 * i +2,:] = Ni - n_hati_rep  # Innovation, N_t - n_t

            for k in range(2 * n + 1):
                N_eval[2 * i: 2 * i + 2, k] = N[2 * i: 2 * i + 2, k] * wc_s[k]
                
            n_hati[1] = pi_to_pi(n_hati[1]) # now use pi_to_pi for angle with respect car of possible landmark
            n_hat[2 * i: 2 * i + 2,:] = n_hati

        # Augmented noise matrix
        R_aug = zeros((dimf * lenidf, dimf * lenidf))
        for i in range(lenidf):
            R_aug[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = self.Re

        # (EQ 10) - Innovation covariance
        S = np.dot(N_eval, np.transpose(N_eval))        # vehicle uncertainty + map + measurement noise
        S = (S + np.transpose(S))*0.5 + R_aug           # make symmetric for better numerical stability

        # Cross covariance: considering vehicle uncertainty
        X = zeros((dimv, 2 * n + 1)) # stack
        for k in range(2 * n + 1):
            ksi_aux = Ksi[:3, k].reshape((3,1))
            X[:,k] = np.squeeze(wc_s[k] * (ksi_aux - self.xv))  # vehicle uncertainty

        # (EQ 11)
        Sigma = np.dot(X , np.transpose(N_eval))            # cross covariance matrix ('dimv' by 'dimf * lenidf')

        # (EQ 12) Kalman gain
        K = Sigma.dot(linalg.inv(S))

        # innovation('dimf*lenidf' by 1)
        v = z - n_hat
        for i in range(lenidf):
            v[2 * i] = pi_to_pi(v[2 * i])

        # (EQ 13) Standard Kƒmax
        # date
        xv = self.xv + K.dot(v)
        # (EQ 14)
        Pv = self.Pv - K.dot(Sigma.T) # same as - Pv1 = self.Pv - linalg.multi_dot([K, S, np.transpose(K)])# CHANGED WITH RESPECT MATLAB IMPLEMENTATION


        # Update weights (parallel process): ERB for SLAM problem
        Lt = S                                             # faster...
        #Lt = (Sigma.T.dot(linalg.inv(Pv))).dot(Sigma) + S   # square matrix of 'dimf*lenidf'
        den = sqrt(2 * pi * linalg.det(Lt))
        num = exp(-0.5 * linalg.multi_dot([np.transpose(v), linalg.inv(Lt), v]))
        w = num / den
        self.w = self.w * w

        # (EQ 15) sample vehicle state from proposal distribution
        xvs = np.random.multivariate_normal(np.squeeze(xv), Pv)
        self.xv = xvs
        self.Pv = eye(3) * EPS # initialize covariance

    def feature_updateu(self):
        """
        Having selected a new pose from the proposal distribution, this pose is
        assumed perfect and each feature update may be computed independently
        and without pose uncertainty.
        Modifies particles xf and Pf.
        """
        R = Particle.Re
        N = Particle.n_f_a
        lmb = Particle.lambda_f_a
        wg_f_a = Particle.wg_f_a
        wc_f_a = Particle.wc_f_a
        dimf = self.zf.shape[1] # feature state dimension
        xf = self.xf[:, self.idf]
        Pf = self.Pf[:,:, self.idf]
        # HARDCODED VALUES JUST FOR TESTING
        #self.xv[0] = 0.169956293382486
        #self.xv[1] = 5.540053567362944e-04
        #self.xv[2] = 1.443584553182200e-04
        for i in range(len(self.idf)):
            # augmented feature state
            xf_aug = np.append(xf[:,i].reshape((2,1)), zeros((2, 1)), axis=0)  # to add control input and observation that agree with known features
            Pf_aug = np.append(
                np.append(Pf[:,:,i], zeros((2, 2)), axis=1),
                np.append(zeros((2, 2)), R, axis=1),
                axis=0
            )
            # disassemble the covariance
            P = (N + lmb) * Pf_aug + EPS * eye(N)
            S = linalg.cholesky(P)
            # get sigma points
            Kai = zeros((N, 2 * N + 1))
            Kai[:, 0] = xf_aug[:, 0]
            for k in range(N):
                Kai[:, k + 1] = xf_aug[:, 0] + S[:, k] # equation 18
                Kai[:, k + 1 + N] = xf_aug[:, 0] - S[:, k] # equation 18
            # transform the sigma points
            Z = zeros((dimf, 2 * N + 1))
            bs = zeros((1, 2 * N + 1)) # bearing sign
            for k in range(2 * N + 1):
                d = Kai[0:2, k] - self.xv[0: 2]
                r = sqrt(d[0]**2 + d[1]**2) + Kai[2, k] # predicted distance
                bearing = atan2(d[1], d[0])
                bs[0,k] = np.sign(bearing)
                if k > 0:  # unify the sign
                    if bs[0,k] != bs[0,k - 1]:
                        if bs[0,k] < 0 and -pi < bearing and bearing < -pi / 2:
                            bearing = bearing + 2 * pi
                            bs[0,k] = np.sign(bearing)
                        elif bs[0,k] > 0 and pi / 2 < bearing and bearing < pi:
                            bearing = bearing - 2 * pi
                            bs[0,k] = np.sign(bearing)
                Z[:, k] = np.append(np.array([r]), np.array([bearing - self.xv[2] + Kai[3,k]]), axis=0) # h(lambda, x) in paper
            z_hat = 0 # predictive observation
            for k in range(2 * N + 1):
                z_hat = z_hat + wg_f_a[0,k] * Z[:,k]
            St = 0 # innovation covariance
            for k in range(2 * N + 1):
                St = St + wc_f_a[0,k] * (Z[:,k] - z_hat).reshape((2,1))*np.transpose((Z[:,k] - z_hat).reshape((2,1)))
            St = (St +St.T) * 0.5 # make symetric
            Sigma = 0 # cross covariance
            for k in range(2 * N + 1):
                Sigma = Sigma + wc_f_a[0,k] * (Kai[0:2, k] - xf[:,i]).reshape((2,1))*np.transpose((Z[:,k] - z_hat).reshape((2,1))) # equation 19
            v = self.zf[i,:] - z_hat
            v[1] = pi_to_pi(v[1])
            # Kalman gain
            Kt = Sigma.dot(linalg.inv(St))
            self.xf[:,self.idf[i]] = xf[:,i] + Kt.dot(v) # equation 20
            self.Pf[:,:,self.idf[i]] = Pf[:,:,i] - (Kt.dot(St)).dot(Kt.T) # equation 21


    def augment_map(self):
        #if len(self.zn) != 0: # new features seen that are not still saved
        if len(self.zf) == 0: # Sample from proposal distribution, if we have not already done so above. Gets inside
            # if already features saved
            self.xv = np.random.multivariate_normal(np.squeeze(self.xv), self.Pv)
            self.Pv = EPS * eye(3)
        self.__add_feature()



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

        self.xf = np.concatenate((self.xf,xf), 1) # concatenate in Feature mean states
        self.Pf = np.concatenate((self.Pf, Pf), 2)

    def deepcopy(self):
        """
        Returns a deep copy of this particle.
        #TO DO: Test it to see if any reference is maintained!
        """
        return copy.deepcopy(self)
