from threading import Thread
class Writer (Thread):
    def __init__(self, message, pipe):
        super(Writer, self).__init__()
        self.message = message
        self.pipe = pipe


    def run(self):
        self.pipe.send(self.message)