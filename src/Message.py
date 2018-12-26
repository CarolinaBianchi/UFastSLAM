"""
Message exchanged between the main process and the plotter process.
"""
class Message():
    def __init__(self, particles, z,  time):
        self.particles = particles
        self.z = z
        self.time = time