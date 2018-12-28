import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
if plt.get_backend()=="MacOSX":
    mp.set_start_method("forkserver")
import Constants as C
import numpy as np
import Message
from math import sin, cos, atan
from matplotlib import collections  as mc


PATH        = "../victoria_park/"
GPS         = "mygps.txt"

alfa = atan(-22/28)
c = cos(alfa)
s = sin(alfa)

class ProcessPlotter (object):
    def __init__(self):
        self.errcount = 0
        self.epath = []
        self.xdata = []
        self.ydata = []
        self.theta = []
        self.xgt = [] #x of ground truth
        self.ygt = [] #y of ground truth
        self.gttime, self.gtx, self.gty = self.init_ground_truth()
        self.line_col = None

        #Initialize figures
        self.fig1, self.ax1 = plt.subplots()
        axes = plt.gca()
        axes.set_xlim(-150, 250)
        axes.set_ylim(-100, 250)

        self.fig2, self.ax2 = plt.subplots()

        self.line, = self.ax1.plot([], [], 'r-')
        self.gt, = self.ax1.plot(self.xgt, self.ygt, 'g-')


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
        plt.close('all')

    def call_back(self):
        """
        Callback of a timer. While polling this method checks if there is any new data to be plotted.
        :return:
        """
        while self.pipe.poll():
            msg = self.pipe.recv()
            if msg is None:
                break
            else:

                self.__plot_epath(msg.particles)
                self.__plot_ground_truth(msg.time)
                #self.__plot_features(msg.particles)
                #self.__plot_laser(msg.z, [self.xdata[-1], self.ydata[-1], self.theta[-1]])
            plt.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        timer = self.fig1.canvas.new_timer(interval=10)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

    def __plot_epath(self, particles):
        """
        Plots the estimated path.
        :param particles:
        :return:
        """
        self.epath = self.__get_epath(self.epath, particles, C.NPARTICLES)
        self.xdata.append(self.epath[0])
        self.ydata.append(self.epath[1])
        self.theta.append(self.epath[2])
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)
        #plt.pause(1e-15)

    def __get_epath(self, epath, particles,  NPARTICLES):
        """
        Gets the estimated path.
        :param epath: path up to this point.
        :param particles:
        :param NPARTICLES:
        :return:
        """
        # vehicle state estimation result
        xvp = [particle.xv for particle in particles]
        w = [particle.w for particle in particles]
        ws = np.sum(w)
        w = w / ws # normalize
        # weighted mean vehicle pose
        xvmean = 0
        for i in range(NPARTICLES):
            contribution = np.squeeze(xvp[i]) # TODO: Check why in a certain step we have particle[0] xvp as 3x1 and the rest 1x3
            xvmean = xvmean + w[i] * contribution
        # keep the pose for recovering estimation trajectory
        return xvmean

    def __plot_features(self, particles):
        """
        Plots the features.
        :param particles:
        :return:
        """
        x = []
        y = []
        ws = [particle.w for particle in particles]
        maxInd = ws.index(max(ws))
        maxP = particles[maxInd]
        #for particle in particles:
        for xf in maxP.xf.T:
            if xf.size:
                x.append(xf[0])
                y.append(xf[1])
        self.ax1.scatter(x, y, s=1, color='black')


    def __plot_ground_truth(self, time):
        """
        Plots the ground truth up to a certain time instant.
        :param time: current time instant.
        """
        i =0
        while(self.gttime[0]<time):
            self.xgt.append(self.gtx[0]*c-self.gty[0]*s)
            self.ygt.append(self.gtx[0]*s+self.gty[0]*c)
            self.gtx.pop(0)
            self.gty.pop(0)
            self.gttime.pop(0)
            i = i+1
        self.ax1.scatter(self.xgt, self.ygt, s=1, color='blue')
        if(i): # horrible.
            self.__plot_error([self.xgt[-1], self.ygt[-1]])


    def __plot_error(self, xv_gps):
        xv_hat = np.array(self.epath)[0:2]
        xv_gps = np.array(xv_gps)

        d = xv_hat-xv_gps
        e = (d[0]*d[0]+d[1]*d[1])**0.5
        self.ax2.scatter(self.errcount, e, color = 'blue')
        self.errcount = self.errcount+1

    def __plot_laser(self, z, xv):
        lines = self.make_laser_lines(z, xv)
        if self.line_col != None : #remove previous laser lines
            self.line_col.remove()
        lc = mc.LineCollection(lines, colors = np.array((0, 1, 0, 1)), linewidths=2)
        self.ax1.add_collection(lc)
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
        # p = line_plot_conversion([lnes_x, lnes_y, lnes_end_pos])
        data = []
        for i in range(len(rb)):
            data.append([(lnes_x[0][i], lnes_y[0][i]), (lnes_end_pos[0][i], lnes_end_pos[1][i])])
            # data.append((lnes_end_pos[0][i], lnes_end_pos[1][i]))
        return data

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





