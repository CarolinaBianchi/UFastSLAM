from Odometry import *
from Particle import *
from Sensor import *
from Vehicle import *
import Constants as C



def init_particles(npart):
    w = 1/npart;
    return [Particle(w) for i in range(npart)]

def resample_particles(particles, N):
    """
    #TO DO
    Resamples the particle if their weight variance is suche that N-effective
    is less than Nmin.
    NB: If the particle is sampled more than once, it is necessary to call the
    deepcopy method to provide an identical object with a different reference!
    In the naive implementation, this method may be called every time (waste of
    time but it may be difficult to find a smart implementation).
    """
    return particles

def main():

    # Initialization
    odom    = OdometryPublisher()
    sensor  = Sensor()
    particles = init_particles(C.NPARTICLES)

    for i,t in enumerate(C.T):
        odomData = odom.read(t)

        if(odomData.speed != 0):
            # Prediction
            for particle in particles:
                particle.predictACFRu(odomData)

            #Measurement
            z = sensor.read(t, C.T[i+1])

            # Data associations
            for particle in particles:
                particle.data_associateNN(z)

            # Known map features
            for particle in particles:
                # Sample from optimal proposal distribution
                particle.sample_proposaluf()

                # Map update
                particle.feature_updateu()

            particles= resample_particles(particles, C.NEFFECTIVE);

            # When new feautres are observed, augment it ot the map
            for particle in particles:
                particle.augment_map()






if __name__ =="__main__":
    main()
