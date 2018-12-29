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
import sys

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
        unique, toCopy = get_unique(keep)
        n_particles = [particles[i] for i in unique]
        for i, v in enumerate(toCopy):
            n_particles.append(particles[v].deepcopy())

        for u in n_particles :
            u.w = 1 / N
        particles = n_particles

    return particles

def get_unique(vals):
    """
    Separates vals into 2 different lists:
    - 1 of the unique values contained in the list
    - repeated values
    The union of the two returned lists is the original one.
    :param self:
    :param array:
    :return:
    """
    unique = []
    repeated = []
    for v in vals:
        if v in unique :
            repeated.append(v)
        else :
            unique.append(v)
    return unique, repeated


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
            z = FrontEnd.filter(sensor.read(t, C.T[i + 1])) # TODO: In case speed at last time is different 0 would crash

            if size(z):
                # Data associations
                for particle in particles:
                    particle.data_associateNN(z)

                # Known map features
                for particle in particles:
                    # Sample from optimal proposal distribution
                    if particle.zf.size:
                        particle.sample_proposaluf()    #Updates state estimation taking into account the measurements (FastSLAM 2.0)

                        # Map update
                        particle.feature_updateu()

                particles = resample_particles(particles, C.NEFFECTIVE)
                plot_pipe.send(Message(particles, z, t))

                # When new feautres are observed, augment it ot the map
                for particle in particles:
                    if particle.zn.size:
                        particle.augment_map()

    plot_pipe.send(True) # some message so the process knows is the end and send the figure
    figure = plot_pipe.recv()
    figure.savefig('results/uslam_map_victoria.png')
    # plot_pipe.join() # Not necessary

if __name__ == "__main__":
    if sys.platform != 'win32': # solve compatibility issues with Mac, TODO: Check if we need to exclude Ubuntu as well
        mp.set_start_method("forkserver")
    main()

