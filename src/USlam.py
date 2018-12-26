from Control import *
from Particle import *
from Sensor import *
from Vehicle import *
from FrontEnd import *
import Constants as C
from Utils import pi_to_pi
from Plot import ProcessPlotter
from Message import Message
import multiprocessing as mp
import time

import matplotlib.pyplot as plt



from math import sin, cos, nan

import numpy as np
from numpy import zeros, eye, size, linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def init_particles(npart):
    w = 1 / npart
    return [Particle(w) for i in range(npart)]


def resample_particles(particles, Nmin):
    """
    #TO DO
    Resamples the particle if their weight variance is suche that N-effective
    is less than Nmin.
    NB: If the particle is sampled more than once, it is necessary to call the
    deepcopy method to provide an identical object with a different reference!
    In the naive implementation, this method may be called every time (waste of
    time but it may be difficult to find a smart implementation).
    """
    N = len(particles)
    w = zeros(N)
    ws = 0
    waux = zeros(N)
    for i in range(N):
        ws = ws + particles[i].w
    for i in range(N):
        waux[i] = particles[i].w / ws
        particles[i].w = waux[i]

    [keep, Neff] = stratified_resample(waux)
    if Neff <= Nmin:
        for i in range(N):
            particles[i] = particles[keep[i]].deepcopy()
            particles[i].w = 1 / N

    return particles


def stratified_resample(w):  # TODO: Check w is normalized
    Neff = 1 / np.sum(np.power(w, 2))

    lenw = len(w)
    keep = zeros(lenw)
    select = stratified_random(lenw)
    w = np.cumsum(w)

    ctr = 0
    for i in range(lenw):
        while ctr < lenw and select[ctr] < w[i]:
            keep[ctr] = i
            ctr = ctr + 1
    return keep.astype(int), Neff # return array of int's, if we remove astype would be array floats


def stratified_random(N):
    k = 1 / N
    #di = np.arange(k / 2, 1 - k / 2, k)  # deterministic intervals
    di = np.arange(k / 2, 1, k)
    s = di + np.random.rand(N) * k - k / 2  # dither within interval
    return s

def make_laser_lines(rb, xv):
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
    lnes_end_pos = TransformToGlobal([np.multiply(lnes_distance[0], np.cos(lnes_angle[0])),
                                   np.multiply(lnes_distance[0], np.sin(lnes_angle[0]))], xv)
    #p = line_plot_conversion([lnes_x, lnes_y, lnes_end_pos])
    data = []
    for i in range(len(rb)):
        data.append([(lnes_x[0][i], lnes_y[0][i]),(lnes_end_pos[0][i], lnes_end_pos[1][i])])
        #data.append((lnes_end_pos[0][i], lnes_end_pos[1][i]))
    return data

def TransformToGlobal(p, b):
    # Transform a list of poses [x;y;phi] so that they are global wrt a base pose
    # rotate
    rot = zeros((2, 2))
    rot[0,0] = cos(b[2])
    rot[0,1] = -sin(b[2])
    rot[1,0] = sin(b[2])
    rot[1,1] = cos(b[2])
    p[0:2] = np.dot(rot, p[0:2])

    # translate
    p[0] = p[0] + b[0]
    p[1] = p[1] + b[1]

    # if p is a pose and not a point
    if len(p) == 3:
       p[2] = pi_to_pi(p[2] + b[2])
    return p


"""
def init_animation():
    # Initialize animation
    x = []
    y = []

def animate(epath):
    y.append(epath[1][1])  # update the data.
    x.append(epath[1][0])  # update the data.

def do_plot(particles, plines, epath):
    xfp = [particle.xf for particle in particles]
    w = [particle.w for particle in particles]
    ii = np.where(w == max(w))[0]
    # TODO: Add animations
"""

def main():
    # Initialization
    ctrl = ControlPublisher()
    sensor = Sensor()
    particles = init_particles(C.NPARTICLES)
    plot_pipe, plotter_pipe = mp.Pipe()
    plotter = ProcessPlotter()
    plot_process = mp.Process(
       target=plotter, args=(plotter_pipe,), daemon=True)
    plot_process.start()
    message = None
    for i, t in enumerate(C.T):

        ctrlData = ctrl.read(t)

        if (ctrlData.speed != 0):
            # Prediction
            for particle in particles:
                particle.predictACFRu(ctrlData)

            # Measurement
            z = FrontEnd.filter(sensor.read(t, C.T[i + 1]))

            if z:
                # TODO: Plot laser lines
                plines = []
                plines = make_laser_lines(z, particles[0].xv) # use the first particle for drawing the laser line

                # Data associations
                for particle in particles:
                    particle.data_associateNN(z)

                # Known map features
                for particle in particles:
                    # Sample from optimal proposal distribution
                    if particle.zf.size:
                        particle.sample_proposaluf()

                        # Map update
                        particle.feature_updateu()

                particles = resample_particles(particles, C.NEFFECTIVE)
                plot_pipe.send(Message(particles, t, plines))

                # When new feautres are observed, augment it ot the map
                for particle in particles:
                    if particle.zn.size:
                        particle.augment_map()

    #plt.show()

if __name__ == "__main__":
    main()

