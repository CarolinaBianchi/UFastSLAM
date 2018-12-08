"""
Odometry.
The main method exposed is the "read" method, which returns the odometry information at time t.
Each time this method is called, an element of type Odometry is returned.
"""


PATH        = "../victoria_park/"
SPEED       = "speed.txt"
STEERING    = "steering.txt"
TIME        = "time.txt"

class Odometry:
    def __init__(self, speed, steering):
        self.speed = speed
        self.steering = steering


class OdometryPublisher:
    def __init__(self):
        self.odometry = self.__read_odometry()

    def __read_odometry(self):
        """
        Populates self.odometry, array of Odometry objects
        """
        odoms={}
        fspeed = open ( PATH + SPEED, 'r')
        fsteer = open ( PATH + STEERING, 'r')
        ftime  = open ( PATH + TIME, 'r')

        odoms = {float(time) : Odometry(float(speed), float(steering)) for time, speed, steering in zip(ftime, fspeed, fsteer)}
        fspeed.close()
        fspeed.close()
        ftime.close()
        return odoms

    def read(self, time):
        """
        Return the odometry information at time t.
        """
        try:
            return self.odometry.pop(time)
        except Exception as e: #no measurements a time t
            return None
odom = OdometryPublisher()
