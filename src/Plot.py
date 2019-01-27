import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from math import sin, cos, atan, pi
from matplotlib import collections  as mc
from scipy.linalg import schur

PATH        = "../victoria_park/"
GPS         = "mygps.txt"
IMGPATH     = "output_img/"
OUTPATH     = "output/"

#Rotational data for GPS
alfa = atan(-0.71)
c = cos(alfa)
s = sin(alfa)
ferr = open(OUTPATH+'error.txt', "w+")
x_map = open(OUTPATH+'x_data.txt', "w+")
y_map = open(OUTPATH+'y_data.txt', "w+")
x_feat = open(OUTPATH+'x_feat.txt', "w+")
y_feat = open(OUTPATH+'y_feat.txt', "w+")
cov = open(OUTPATH+'cov.txt', "w+")

class ProcessPlotter (object):
    def __init__(self):
        self.errcount = 0
        self.err_vect = []
        self.error_value = []
        self.path_count = 0
        self.err = []
        self.epath = []
        self.xdata = []
        self.ydata = []
        self.theta = []
        self.xgt = [] #x of ground truth
        self.ygt = [] #y of ground truth
        self.covariances = []
        self.gttime, self.gtx, self.gty = self.init_ground_truth()
        self.line_col = None

        # Create output directories
        import os
        if not os.path.exists(OUTPATH):
            os.makedirs(OUTPATH)
        if not os.path.exists(IMGPATH):
            os.makedirs(IMGPATH)

    def settings(self, do_plot, frequency, save):
        """Enables plotting and its frequency and saving images to output folder."""
        self.doplot = do_plot
        self.frequency = frequency
        self.save = save
        if(do_plot):

            self.figtot, self.axtot = plt.subplots()
            self.axtot.set_xlim(-150, 250)
            self.axtot.set_ylim(-150, 250)
            self.axtot.set_aspect('equal')
            self.axtot.set_xlabel('x [m]')
            self.axtot.set_ylabel('y [m]')
            self.axtot.set_title('Result map against GT')

            self.figerr, self.axerr = plt.subplots()
            self.axerr.set_ylim(0, 17)
            self.axerr.set_xlabel('number steps')
            self.axerr.set_xlabel('number steps')
            self.axerr.set_ylabel('error [m]')
            self.axerr.set_title('Error')

            self.line, = self.axtot.plot([], [], 'r-')

            self.gt, = self.axtot.plot(self.xgt, self.ygt, 'g-')
            self.oldFeatures = self.axtot.scatter([], [])
            self.olderror = self.axerr.scatter([], [])

            plt.ion()
            #mng = plt.get_current_fig_manager()
            #mng.window.state('zoomed')
            plt.show()

    def init_ground_truth(self):
        """
        Processes the GPS data and rotates it in order to match the control data rotation.
        :return:
        """
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
        """
        Closes all, saves last figure.
        """
        ferr.close()
        x_map.close()
        y_map.close()

        self.figtot.savefig(IMGPATH+'/uslam_map_victoria.png')
        plt.close('all')

    def update(self, msg):
        """
        Updates the plot
        :param msg: message containing the new information produced in this step.
        :return:
        """
        self.epath = msg.xv
        self.xdata.append(self.epath[0])
        self.ydata.append(self.epath[1])
        self.theta.append(self.epath[2])
        self.__plot_ground_truth(msg.time, msg)
        if(self.doplot):
            plt.draw()
            plt.pause(1e-15)

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

        if(self.doplot):
            if self.errcount % self.frequency == 0:
                self.axtot.scatter(self.xgt, self.ygt, s=1, color='blue')

            if(i): # If we got a new ground truth info.
                self.__plot_error([self.xgt[-1], self.ygt[-1]], msg)

    def save_feat(self, particles):
        """
        Saves estimated feature position to a file.
        :param particles:
        :return:
        """
        ws = [particle.w for particle in particles]
        maxInd = ws.index(max(ws))
        maxP = particles[maxInd]
        for i in len(maxP.xf):
            x_feat.write("%f\n" % maxP.xf[i][0])
            y_feat.write("%f\n" % maxP.xf[i][0])
            cov.write("%f,%f,%f,%f\n" % maxP.Pf[i][0], maxP.Pf[i][0], maxP.Pf[i][0], maxP.Pf[i][0])
        x_feat.close()
        y_feat.close()
        cov.close()

    def __plot_epath(self, xv):
        """
        Plots the estimated path.
        :param particles:
        :return:
        """
        self.epath = xv
        self.xdata.append(self.epath[0])
        self.ydata.append(self.epath[1])
        self.theta.append(self.epath[2])
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

    def plot_error(self, e):
        """Plots the error in estimation."""
        self.olderror.remove()
        y = []
        for f in self.error_value:
            y.append(f)
        self.olderror = self.axerr.scatter(self.err_vect, y, s=1, color='black')


    def __plot_error(self, xv_gps, msg):
        xv_hat = np.array(self.epath)[0:2]
        xv_gps = np.array(xv_gps)

        d = xv_hat-xv_gps
        e = (d[0]*d[0]+d[1]*d[1])**0.5

        self.err_vect.append(self.errcount)
        self.error_value.append(e)

        if self.errcount % self.frequency == 0 and self.doplot:
            self.__plot_epath(msg.xv)
            self.__plot_laser(msg.z, msg.xv)
            self.plot_error(e)
            if len(msg.xf):
                self.__plot_features(msg.xf)
                self.__plot_covariance_ellipse(msg.xv, msg.Pv, msg.xf, msg.Pf)
            if(self.save):
                s = IMGPATH+'map' + str(self.errcount) + '.png'
                p = IMGPATH+'error' + str(self.errcount) + '.png'
                self.figtot.savefig(s)
                self.figerr.savefig(p)

        self.errcount = self.errcount + 1
        x_map.write("%f\n" %self.xdata[0])
        y_map.write("%f\n" %self.ydata[0])
        ferr.write("%f\n" %e)


    def __plot_laser(self, z, xv):
        """
        Plots laser lines.
        :param z: observed features
        :param xv: vehicle pose.
        """
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

            # lnes = np.append([lnes_x, lnes_y, lnes_angle], axis = 0)
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
         self.make_ellipse(xv, Pv, xf, Pf, circ)

    def make_ellipse(self, xv, Pv, xf, Pf, circ): # if integ 0 plot cov of vehicle, if 1 landmark covariance
        for c in self.covariances:
            c.remove()
        self.covariances = []
        cov_veh_plot = self.obtain_squared_P(Pv[:2,:2], circ, xv[:2]) # plot the covariance of the vehicle position
        self.covariances.append(cov_veh_plot) # to remove it when redrawing
        for i in range(np.size(xf, 1)):  # number of known features
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
        position = np.squeeze(pos)
        position = position.reshape((2, 1))
        p = a + np.matlib.repmat(position, 1, a.shape[1])
        p1 = self.axtot.scatter(p[0], p[1], s=1, color='green')
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