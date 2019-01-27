from Control import *
from Particle import *
from Sensor import *
from Vehicle import *
from FrontEnd import *
from Plot import ProcessPlotter
from Message import Message
import multiprocessing as mp
import sys
import numpy as np
from numpy import zeros, eye, size, linalg
import argparse

SAVE = False # set to save or not output figures
DOPLOT = True # set to plot
FREQUENCY = 50 # default plotting frequency

epath = []
def main():

    # Initialization
    ctrl = ControlPublisher()
    sensor = Sensor()


    particles = init_particles(C.NPARTICLES)
    plotter = ProcessPlotter()
    plotter.settings(DOPLOT, FREQUENCY, SAVE)

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
                plotter.update(get_message(particles, z, t))

                # When new feautres are observed, augment it ot the map
                for particle in particles:
                    if particle.zn.size:
                        particle.augment_map()

    plotter.terminate()

def init_particles(npart):
    """
    Initializes the particle set with npart particles. The weights are initialized as 1/npart.
    :param npart: number of particles
    :return: list of particles.
    """
    w = 1 / npart
    return [Particle(w) for i in range(npart)]


def resample_particles(particles, Nmin):
    """
    Resamples the particle if their weight variance is such that N-effective
    is less than Nmin.
    NB: If the particle is sampled more than once, it is necessary to call the
    deepcopy method to provide an identical object with a different reference!
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
    :param vals: a list of integers.
    :return: unique, repeated. unique is the list of unique values in vals. unique union repeated = vals.
    """
    unique = []
    repeated = []
    for v in vals:
        if v in unique :
            repeated.append(v)
        else :
            unique.append(v)
    return unique, repeated


def stratified_resample(w):
    """
    Performs a round of stratified resampling.
    :param w: the weights of the particles
    :return: keep, Neff. keep is an array of indexes of the particles to keep, Neff is a measure of the sparsity of the particle weights.
    """
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
    """
    Returns N equally spaced numbers in the range [0, 1]
    :param N:
    :return: array
    """
    k = 1 / N
    di = np.arange(k / 2, 1, k)
    s = di + np.random.rand(N) * k - k / 2  # dither within interval
    return s


def get_message(particles, z, t):
    """
    Returns a Message object with the needed information for the plotter to update the figure.
    :param particles: particle set
    :param z: current measure
    :param t: current time instant
    :return:
    """
    # Estimated path:
    ws = [particle.w for particle in particles]
    maxInd = ws.index(max(ws))
    maxP = particles[maxInd]
    xv, Pv = __get_epath(particles, maxInd)
    xf, Pf = __get_features(maxP)
    return Message(xv, Pv, xf, Pf, z, t)

def __get_features(maxP):
    """
    Retrieves the estimated feature position from the particle with the maximum weight.
    :param maxP: particle with maximum weight.
    :return:
    """
    f, P = [],[]
    if maxP.xf.T.size:
        f = [xf for xf in maxP.xf.T]
        P = [Pf for Pf in maxP.Pf.T]
    return f, P

def __get_epath(particles, maxInd):
    """
    Gets the estimated path from the particle with maximum weight.
    :param particles:
    :return:
    """
    global epath
    # vehicle state estimation result
    xvp = [particle.xv for particle in particles]
    w = [particle.w for particle in particles]
    ws = np.sum(w)
    w = w / ws  # normalize
    # weighted mean vehicle pose
    xvmean = 0
    for i, part in enumerate(particles):
        contribution = np.squeeze(
            xvp[i])
        xvmean = xvmean + w[i] * contribution
    # keep the pose for recovering estimation trajectory
    epath.append(xvmean)
    return xvmean, particles[maxInd].Pv

def get_message(particles, z, t):
    """Returns the essential informations to update the plot."""
    # Estimated path:
    ws = [particle.w for particle in particles]
    maxInd = ws.index(max(ws))
    maxP = particles[maxInd]
    xv, Pv = __get_epath(particles, maxInd)
    xf, Pf = __get_features(maxP)
    return Message(xv, Pv, xf, Pf, z, t)

def __get_features(maxP):
    """Returns the features poses and covariances as estimated from the most likely particle."""
    f, P = [],[]
    for xf in maxP.xf.T:
        if xf.size:
            f.append(xf)
    for Pf in maxP.Pf:
        if Pf.size:
            P.append(Pf)
    return f, P

def __get_epath(particles, maxInd):
    """
    Gets the estimated path.
    :param particles:
    :return:
    """
    global epath
    # vehicle state estimation result
    xvp = [particle.xv for particle in particles]
    w = [particle.w for particle in particles]
    ws = np.sum(w)
    w = w / ws  # normalize
    # weighted mean vehicle pose
    xvmean = 0
    for i, part in enumerate(particles):
        contribution = np.squeeze(
            xvp[i])
        xvmean = xvmean + w[i] * contribution
    # keep the pose for recovering estimation trajectory
    epath.append(xvmean)
    return xvmean, particles[maxInd].Pv

def __set_plot(value):
    """
    Sets the option of plotting.
    :param value: plotting frequency. The plot is turned off if value is <=0.
    """
    global DOPLOT, FREQUENCY
    DOPLOT = True
    FREQUENCY = 40 # default frequency
    if value:
        DOPLOT = value > 0
        FREQUENCY = value

    print("do plot "+str(DOPLOT))

def __set_save(value):
    """
    Sets the option to save the output.
    """
    global SAVE
    SAVE = value

if __name__ == "__main__":
    if sys.platform != 'win32': # solve compatibility issues with Mac
        mp.set_start_method("forkserver")

    parser = argparse.ArgumentParser(description='Implementation of Unscented Fast SLAM algorithm.')
    parser.add_argument('-p', dest='plot', type=int, nargs='?',
                        help='plot option. If the value specified is < = 0, then the nothing will be plotted, \n'
                             'otherwise the number determines the frequency of plot updates (lower = higher frequency).')
    parser.add_argument('-s', dest='save', action='store_true',
                        help='if specified, saves all the producted plots in the /output folder')

    args = parser.parse_args()
    print(args)
    __set_save(args.save)
    __set_plot(args.plot)
    main()