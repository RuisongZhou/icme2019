# -*- coding : utf-8 -*-

import sys

class Logger(object):
    def __init__(self, filename='log.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self,  message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
