#from multiprocessing import queue
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import Constants as C
import numpy as np

class ProcessPlotter (object):
    def __init__(self):
        self.epath = []
        self.xdata = []
        self.ydata = []
        axes = plt.gca()
        axes.set_xlim(-150, 250)
        axes.set_ylim(-100, 250)
        self.line, = axes.plot(self.xdata, self.ydata, 'r-')
        print(plt.get_backend())
        #plt.show()


    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            data = self.pipe.recv()
            if data is None:
                break
                #self.terminate()
                #return False
            else:
                self.__plot_epath(data)
                self.__plot_features(data)
            plt.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        axes = plt.gca()
        axes.set_xlim(-150, 250)
        axes.set_ylim(-100, 250)
        timer = self.fig.canvas.new_timer(interval=5)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        #plt.interactive(False)
        self.line, = plt.gca().plot([], [], 'r-')
        plt.show()

    def __plot_epath(self, particles):
        self.epath = self.__get_epath(self.epath, particles, C.NPARTICLES)
        self.xdata.append(self.epath[0])
        self.ydata.append(self.epath[1])
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)
        #plt.pause(1e-15)

    def __get_epath(self, epath, particles,  NPARTICLES):
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
        x = []
        y = []
        for particle in particles:

            for xf in particle.xf.T:
                if xf.size:
                    x.append(xf[0])
                    y.append(xf[1])
        plt.scatter(x, y, s=1, color='blue')
        #plt.draw()
        #plt.pause(1e-15)


