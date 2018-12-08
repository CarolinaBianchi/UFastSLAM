"""
Front end of UFastSLAM .
Elaborates raw data coming from sensors.
"""
from Constants import perceplimit as LIMIT, PI as pi

class FrontEnd:
    def filter(listMeasurements):
        """
        Filters a list of measurements, returing only the valid ones.
        """
        if (len(listMeasurements) == 0):
            return listMeasurements

        filtered = [[meas for meas in l.list if meas.distance!=0 and meas.distance <LIMIT ] for l in lisMeasurements]
        #TO DO!!!! meas.angle= meas.angle - pi/2
        return filtered
