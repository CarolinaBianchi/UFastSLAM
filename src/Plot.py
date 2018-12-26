import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
if plt.get_backend()=="MacOSX":
    mp.set_start_method("forkserver")
import Constants as C
import numpy as np
import Message
from math import sin, cos
from matplotlib import collections  as mc


PATH        = "../victoria_park/"
GPS         = "GPS.txt"
GPS_X       = "gps_x.txt"
GPS_Y       = "gps_y.txt"
MYGPS       = "mygps.txt"

alfa = -19.0/28.0
rot = [[cos(alfa), -sin(alfa)],
       [sin(alfa), cos(alfa)]]
c = cos(alfa)
s = sin(alfa)
x = [-67.649271,-41.714218]
x0 = [x[0]*c-x[1]*s, x[0]*s+x[1]*c]
class ProcessPlotter (object):
    def __init__(self):
        self.epath = []
        self.xdata = []
        self.ydata = []
        self.xgt = [] #x of ground truth
        self.ygt = [] #y of ground truth
        self.gttime, self.gtx, self.gty = self.init_ground_truth()
        self.line_col = []


    def init_ground_truth(self):
        f = open(PATH+MYGPS)
        data = [[float(num) for num in line.split(',') if len(num) > 0] for line in f]
        f.close()
        t, x, y = [],[],[]
        toff, xoff, yoff = data[0][0], data[0][1], data[0][2]
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
                self.__plot_laser(msg.laser)
            plt.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        axes = plt.gca()

        axes.set_xlim(-150, 250)
        axes.set_ylim(-100, 250)
        self.line, = axes.plot(self.xdata, self.ydata, 'r-')
        self.gt, = axes.plot(self.xgt, self.ygt, 'g-')
        timer = self.fig.canvas.new_timer(interval=20)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        self.line, = plt.gca().plot([], [], 'r-')
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
        for particle in particles:
            for xf in particle.xf.T:
                if xf.size:
                    x.append(xf[0])
                    y.append(xf[1])
        plt.scatter(x, y, s=1, color='blue')

    def __plot_ground_truth(self, time):
        """
        Plots the ground truth up to a certain time instant.
        :param time: current time instant.
        """
        while(self.gttime[0]<time):
            self.xgt.append(self.gtx[0]*c-self.gty[0]*s)
            self.ygt.append(self.gtx[0]*s+self.gty[0]*c)
            self.gtx.pop(0)
            self.gty.pop(0)
            self.gttime.pop(0)
        plt.scatter(self.xgt, self.ygt, s=1, color='blue')

    def __plot_laser(self, lines):
        # TODO: Remove the laser lines from the previous period
        # self.line_col.remove()
        lc = mc.LineCollection(lines, colors = np.array((0, 1, 0, 1)), linewidths=2)
        self.ax.add_collection(lc)
        self.line_col = lc




