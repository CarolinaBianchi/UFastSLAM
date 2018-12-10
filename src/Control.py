"""
Control.
The main method exposed is the "read" method, which returns the control signals at time t.
Each time this method is called, an element of type Odometry is returned.
"""


PATH        = "../victoria_park/"
SPEED       = "speed.txt"
STEERING    = "steering.txt"
TIME        = "time.txt"

class Control:
    def __init__(self, speed, steering):
        self.speed = speed
        self.steering = steering


class ControlPublisher:
    def __init__(self):
        self.ctrls = self.__read_controls()

    def __read_controls(self):
        """
        Populates self.odometry, array of Odometry objects
        """
        ctrls={}
        fspeed = open ( PATH + SPEED, 'r')
        fsteer = open ( PATH + STEERING, 'r')
        ftime  = open ( PATH + TIME, 'r')

        ctrls = {float(time) : Control(float(speed), float(steering)) for time, speed, steering in zip(ftime, fspeed, fsteer)}
        fspeed.close()
        fspeed.close()
        ftime.close()
        return ctrls

    def read(self, time):
        """
        Return the control information at time t.
        """
        try:
            return self.ctrls.pop(time)
        except Exception as e: #no measurements a time t
            return None
