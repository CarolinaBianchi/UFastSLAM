from Control import *
from Particle import *
from Sensor import *
from Vehicle import *
import Constants as C


def init_particles(npart):
    w = 1 / npart;
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
        particles = particles[keep]
        for i in range(N):
            particles[i] = 1 / N

    return particles


def stratified_resample(w):  # TODO: Check w is normalized
    Neff = 1 / np.sum(np.power(w, 2))

    lenw = len(w)
    keep = zeros(lenw)
    select = stratified_random(lenw)
    w = np.cumsum(w)

    ctr = 0
    for i in range(lenw):
        while ctr < lenw and select[ctr] < w[i]
            keep[ctr] = i
            ctr = ctr + 1
    return keep, Neff


def stratified_random(N):
    k = 1 / N
    di = np.arange(k / 2, 1 - k / 2, k)  # deterministic intervals
    s = di + np.random.rand(N) * k - k / 2  # dither within interval
    return s


def main():
    # Initialization
    ctrl = ControlPublisher()
    sensor = Sensor()
    particles = init_particles(C.NPARTICLES)
    for i, t in enumerate(C.T):
        ctrlData = ctrl.read(t)

        if (ctrlData.speed != 0):
            # Prediction
            for particle in particles:
                particle.predictACFRu(ctrlData)

            # Measurement
            z = sensor.read(t, C.T[i + 1])

            # Data associations
            for particle in particles:
                particle.data_associateNN(z)

            # Known map features
            for particle in particles:
                # Sample from optimal proposal distribution
                particle.sample_proposaluf()

                # Map update
                particle.feature_updateu()

            particles = resample_particles(particles, C.NEFFECTIVE);

            # When new feautres are observed, augment it ot the map
            for particle in particles:
                particle.augment_map()


if __name__ == "__main__":
    main()

