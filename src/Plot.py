import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Constants as C
import numpy as np
from numpy import linalg
from math import sin, cos, atan, pi
from matplotlib import collections  as mc
from scipy.linalg import schur
PATH        = "../victoria_park/"
GPS         = "mygps.txt"

alfa = atan(-0.71)
c = cos(alfa)
s = sin(alfa)
ferr = open('output/error.txt', "w+")
np.random.seed(1)

class ProcessPlotter (object):
    def __init__(self):
        self.errcount = 0
        self.err = []
        self.errvec = []
        self.epath = []
        self.xdata = []
        self.ydata = []
        self.theta = []
        self.xgt = [] #x of ground truth
        self.ygt = [] #y of ground truth
        self.covariances = []
        self.gttime, self.gtx, self.gty = self.init_ground_truth()
        self.line_col = None

        #Initialize figures
        self.figtot, self.axtot = plt.subplots()
        self.axtot.set_xlim(-150, 250)
        self.axtot.set_ylim(-150, 250)
        self.axtot.set_xlabel('x [m]')
        self.axtot.set_ylabel('y [m]')
        self.axtot.set_title('Result map against GT', fontsize = 18)

        self.figerr, self.axerr = plt.subplots()
        self.axerr.set_ylim(0, 10)
        self.axerr.set_xlabel('number steps')
        self.axerr.set_ylabel('error [m]')
        self.axerr.set_title('Error', fontsize = 18)

        self.line, = self.axtot.plot([], [], 'r-')
        self.gt, = self.axtot.plot(self.xgt, self.ygt, 'g-')
        self.oldFeatures = self.axtot.scatter([],[])

        plt.ion()
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # works fine on Windows!
        plt.show()

    def init_ground_truth(self):
        f = open(PATH+GPS)
        data = [[float(num) for num in line.split(',') if len(num) > 0] for line in f]
        f.close()
        t, x, y = [],[],[]
        xoff, yoff =  data[0][1], data[0][2]
        for d in data:
            t.append(d[0])
            x.append(d[1]-xoff)
            y.append(d[2]-yoff)
        return t, x, y

    def terminate(self):
        ferr.close()
        self.figtot.savefig('output/uslam_map_victoria.png')
        self.figerr.savefig('output/uslam_map_victoria_error.png')

        plt.close('all')

    def update(self, msg):
        self.epath = msg.xv
        self.xdata.append(self.epath[0])
        self.ydata.append(self.epath[1])
        self.theta.append(self.epath[2])
        self.__plot_ground_truth(msg.time, msg)

        plt.draw()
        plt.pause(1e-15)

    def __plot_epath(self, xv):
        """
        Plots the estimated path.
        :param particles:
        :return:
        """
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)

    def __plot_features(self, xf):
        """
        Plots the features.
        :param particles:
        :return:
        """
        self.oldFeatures.remove()
        x, y = [], []
        for f in xf:
            if len(xf):
                x.append(f[0])
                y.append(f[1])
        self.oldFeatures = self.axtot.scatter(x, y, s=1, color='black')


    def __plot_ground_truth(self, time, msg):
        """
        Plots the ground truth up to a certain time instant.
        :param time: current time instant.
        """
        i = 0
        while(self.gttime[0]<time):
            self.xgt.append(self.gtx[0]*c-self.gty[0]*s)
            self.ygt.append(self.gtx[0]*s+self.gty[0]*c)
            self.gtx.pop(0)
            self.gty.pop(0)
            self.gttime.pop(0)
            i = i+1
        if self.errcount % 40 == 0:
            self.axtot.scatter(self.xgt, self.ygt, s=1, color='blue')
        if(i): # horrible.
            self.__plot_error([self.xgt[-1], self.ygt[-1]], msg)


    def __plot_error(self, xv_gps, msg):
        xv_hat = np.array(self.epath)[0:2]
        xv_gps = np.array(xv_gps)

        d = xv_hat-xv_gps
        e = (d[0]*d[0]+d[1]*d[1])**0.5
        self.err.append(e)
        self.errvec.append(self.errcount)
        if self.errcount % 40 == 0:
            self.axerr.scatter(self.errvec, self.err, color='black', s=1)
            self.__plot_epath(msg.xv)
            self.__plot_laser(msg.z, msg.xv)
            if len(msg.xf):
                self.__plot_features(msg.xf)
                self.__plot_covariance_ellipse(msg.xv, msg.Pv, msg.xf, msg.Pf)
            s = 'output/map' + str(self.errcount) + '.png'
            a = 'output/error' + str(self.errcount) + '.png'

            self.figtot.savefig(s)
            self.figerr.savefig(a)


        self.errcount = self.errcount + 1
        ferr.write("%f\n" %e)


    def __plot_laser(self, z, xv):
        lines = self.make_laser_lines(z, xv)
        if self.line_col != None : #remove previous laser lines
            self.line_col.remove()
        lc = mc.LineCollection(lines, colors = np.array(('yellow', 'yellow', 'yellow', 'yellow')), linewidths=2)
        self.axtot.add_collection(lc)
        self.line_col = lc

    def make_laser_lines(self, rb, xv):
        """
        Creates the laser lines from the estimated position of the car to the detected obstacles.
        :param rb:
        :param xv: vehicle position
        :return: list of laser lines
        """
        if not rb:
            p = []
            return
        len_ = len(rb)
        lnes_x = np.zeros((1, len_)) + xv[0]
        lnes_y = np.zeros((1, len_)) + xv[1]
        lnes_distance = np.zeros((1, len_))
        lnes_angle = np.zeros((1, len_))
        # TODO: Check rb structure
        for i in range(len(rb)):
            lnes_distance[0][i] = rb[i].distance
            lnes_angle[0][i] = rb[i].angle

        lnes_end_pos = self.TransformToGlobal([np.multiply(lnes_distance[0], np.cos(lnes_angle[0])),
                                          np.multiply(lnes_distance[0], np.sin(lnes_angle[0]))], xv)
        data = []
        for i in range(len(rb)):
            data.append([(lnes_x[0][i], lnes_y[0][i]), (lnes_end_pos[0][i], lnes_end_pos[1][i])])
        return data

    def __plot_covariance_ellipse(self, xv, Pv, xf, Pf):
        N = 10 # number of points ellipse
        phi = np.array(np.linspace(0, 2 * pi, N, endpoint=True))
        circ = 2 * np.array([np.cos(phi), np.sin(phi)])
        for c in self.covariances:
            c.remove()
        self.covariances = []
        cov_veh_plot = self.obtain_squared_P(Pv[:2,:2], circ, xv[:2]) # plot the covariance of the vehicle position
        self.covariances.append(cov_veh_plot) # to remove it when redrawing
        for i in range(len(xf)):  # number of known features
            cov_feat_plot = self.obtain_squared_P(Pf[i][:2, :2], circ, xf[i][:2]) # plot the covariance of the features
            self.covariances.append(cov_feat_plot)  # to remove it when redrawing

    def obtain_squared_P(self, P, circ, pos):
        """
        :param P:
        :param circ:
        :param pos:
        :return:
        Obtain the radius of the covariance after squaring the matrix and plotting it. Both position vehicle and
        position feature covariances are plotted
        """
        R = np.zeros((2, 2))
        [T, Q] = schur(P)
        R[0, 0] = np.sqrt(T[0, 0])
        R[1, 1] = np.sqrt(T[1, 1])
        R[0, 1] = T[0, 1] / (R[0, 0] + R[1, 1])
        r = linalg.multi_dot([Q, R, Q.T])
        a = np.dot(r, circ)
        position = np.squeeze(pos) # TODO: Check why sometimes xv has dimension (2,1) and sometimes (2,)
        position = position.reshape((2, 1))
        p = a + np.matlib.repmat(position, 1, a.shape[1])
        p1 = self.axtot.scatter(p[0], p[1], s=1, color='green') # TODO: Check if you can provide array vectors instead of integers
        return p1

    def TransformToGlobal(self, p, b):
        # Transform a list of poses [x;y;phi] so that they are global wrt a base pose
        # rotate
        phi = b[2]
        rot = np.array([[cos(phi), -sin(phi)],
               [sin(phi), cos(phi)]])
        p[0:2] = np.dot(rot, p[0:2])

        # translate
        p[0] = p[0] + b[0]
        p[1] = p[1] + b[1]

        return p





