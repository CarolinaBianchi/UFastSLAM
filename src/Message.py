"""
Message exchanged between the main process and the plotter process.
"""
class Message():
    def __init__(self, particles, time):
        self.particles = particles
        self.time = time