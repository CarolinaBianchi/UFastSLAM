from Control import *
from Particle import *
from Sensor import *
from Vehicle import *
from FrontEnd import *
import Constants as C
from Utils import pi_to_pi

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
    # TODO: Check rb structure
    lnes_angle = TransformToGlobal([rb[0]*np.array(cos(rb[1,:])), rb[0,:]*np.array(sin(rb[1,:]))], xv)
    lnes = np.append([lnes_x, lnes_y, lnes_angle], axis = 0)
    p = line_plot_conversion(lnes)

def TransformToGlobal(p, b):
    # Transform a list of poses [x;y;phi] so that they are global wrt a base pose
    # rotate
    rot = zeros((2, 2))
    rot[0,0] = cos(b[2])
    rot[0,1] = -sin(b[2])
    rot[1,0] = sin(b[2])
    rot[1,1] = cos(b[2])
    p[0:2,:] = rot*p[0:2,:]

    # translate
    p[0,:] = p[0,:] + b[0]
    p[1,:] = p[1,:] + b[1]

    # if p is a pose and not a point
    if p.shape[0] == 3:
       p[2,:] = pi_to_pi(p[2,:] + b[2])
    return p

def line_plot_conversion(lne):
    """
    INPUT: list of lines[x1; y1; x2; y2]
    OUTPUT: list of points[x; y]
    ----------------------------------
    Convert a list of lines so that they will be plotted as a set of unconnected lines but only require a single handle
    to do so. This is performed by converting the lines to a set of points, where a NaN point is inserted between
    every point-pair
    """
    len = lne.shape[1] * 3 - 1
    p = zeros((2, len))

    p[:, ::3] = lne[0: 2,:]
    p[:,1::3] = lne[2: 4,:]
    p[:, 2::3] = nan
    return p

def get_epath(particles, epath, NPARTICLES):
    # vehicle state estimation result
    xvp = [particle.xv for particle in particles]
    w = [particle.w for particle in particles]
    ws = np.sum(w)
    w = w / ws # normalize

    # weighted mean vehicle pose
    xvmean = 0
    for i in range(NPARTICLES):
        xvmean = xvmean + w[i] * xvp[i]

    # keep the pose for recovering estimation trajectory
    return [epath, xvmean]

def init_animation():
    # Initialize animation


def do_plot(particles, plines, epath):
    xvp = [particle.xv for particle in particles]
    xfp = [particle.xf for particle in particles]
    w = [particle.w for particle in particles]
    ii = np.where(w == max(w))[0]
    # TODO: Add animations
    fig = plt.figure()
    fig.add_axes([-150, 250, -100, 250])
    fig.xlabel('[m]')
    fig.ylabel('[m]')
    ani = animation.FuncAnimation(fig, animate, init_func=init_animation, interval=2, blit=True, save_count=50)

    if xvp:
        set(h.xvp, 'xdata', xvp[0,:], 'ydata', xvp[1,:])
    if xfp:
        set(h.xfp, 'xdata', xfp[0,:], 'ydata', xfp[1,:])
    if plines:
        set(h.obs, 'xdata', plines(1,:), 'ydata', plines(2,:))
    pcov = make_covariance_ellipses(particles(ii(1)));
    if pcov:
        set(h.cov, 'xdata', pcov(1,:), 'ydata', pcov(2,:))
    set(h.epath, 'xdata', epath(1,:), 'ydata', epath(2,:))
    drawnow

def main():
    # Initialization
    ctrl = ControlPublisher()
    sensor = Sensor()
    particles = init_particles(C.NPARTICLES)
    epath = []
    plines = zeros((2, 1))
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

                # When new feautres are observed, augment it ot the map
                for particle in particles:
                    if particle.zn.size:
                        particle.augment_map()

            #epath = get_epath(particles, epath, C.NPARTICLES)
            #do_plot(particles, plines, epath);


if __name__ == "__main__":
    main()

