"""
Class representing the distance sensor.
The main method exposed is the "read" method.
Each time this method is called, the sensor returns the set of read values at
the current time instantself.
"""

PATH        = "../victoria_park/"
DISTANCE    = "ObservationDistance.txt"
ANGLE       = "ObservationAngle.txt"
LASER_SAMP_T= "lasersampling.txt"        # 7249 elements
TIMESTEPS   = "time.txt"                 # 61945 elements

N_MEASUREMENTS = 0
N_SAMPLING = 0

class SensorMeasurement:
    """
    Distance and angle of the detected object.
    """
    def __init__(self, distance, angle):
        self.distance = distance
        self.angle = angle

class ListSensorMeasurements:
    """
    Collection of measurements taken at time t. List is a list of objects of
    type SensorMeasurement.
    """
    def __init__(self, time, list):
        self.time = time
        self.list = list

class Sensor:
    # Reads the ANGLE/DISTANCE file, creates a List of objects of type ListSensorMeasurements.
    def __init__(self):
        self.measurements = self.__read_measurements()

    def __read_measurements(self):
        """
        Reads the measurements file and populates self.measurements.
        """
        f = open ( PATH+DISTANCE, 'r')
        d = [[float(num) for num in line.split('  ') if len(num)>0] for line in f ]
        f.close()
        f = open(PATH + ANGLE)
        a = [[float(num) for num in line.split('  ') if len(num)>0] for line in f ]
        f.close()
        f = open(PATH + LASER_SAMP_T)
        t = [float(line) for line in f]
        f.close()
        measurements = [ListSensorMeasurements(time, [SensorMeasurement(dist, ang) for dist, ang in zip(distances, angles)]) for distances, angles, time in zip(d, a, t)]

        global N_MEASUREMENTS, N_SAMPLING
        N_MEASUREMENTS = len(measurements)
        N_SAMPLING = len(t)
        #for distances,angles, time  in zip(d, a,t):
        #    measurement_dt = []
        #    for dist, ang in zip(distances, angles):
        #        measurement_dt.append(SensorMeasurement(dist, ang))
        #    measurements.append(ListSensorMeasurements(t, measurement_dt))


        return measurements


    def read(self, t1, t2):
        """
        Returns the list of ListSensorMeasurement taken between t1 and t2.
        If no measurements were made between t1 and t2, returns an empty list.
        """
        l = []

        try: # Faster (I suppose) than checking explicitly the current size of the list
            while(True):
                next = self.measurements[0]   # throws an exception if the list is empty

                if(next.time < t1): #Old measurement that was not read (I don't believe it is possible)
                    self.measurements.pop(0)
                    continue

                if(next.time >= t1 and next.time<t2):
                    # TODO: As always appends 18 measurements per timestep even though some of the have distance 0 and angle < 10^6
                    for i in range(len(next.list)):
                        if next.list[i].distance!=0:
                            l.append(next.list[i])
                    self.measurements.pop(0)
                else:
                    break
        except: # no more measurements
            pass
        return l

s = Sensor()
#print(s.read(24300, 28440))
