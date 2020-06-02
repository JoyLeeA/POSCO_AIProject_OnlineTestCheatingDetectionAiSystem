#   Timer
#   Modified by Jongha
#   Last Update: 2020.06.02

from __future__ import print_function
import time
import numpy as np

class Timer():
    
    def __init__(self, update=True):
        self.stage, self.start = {}, {}
        self.update = update
        self.start_anonymous = time.time() * 1000

    def tic(self, name=None):
        if name: 
            self.start[name] = time.time() * 1000
        else: 
            self.start_anonymous = time.time() * 1000

    def toc(self, name=None):
        if name in self.start.keys():
            period = time.time() * 1000 - self.start[name]
            
            if name not in self.stage.keys(): self.add_stage(name)
            self.update_min_max(name, period)
        
        else:
            period = time.time() * 1000 - self.start_anonymous
        
        return period

    def add_stage(self, name):
        if name not in self.stage.keys():
            self.stage[name] = {'min': np.inf, 'max': 0, 'avg': 0}

    def del_stage(self, name):
        self.stage.pop(name, None)

    def enable_update(self):
        self.update = True

    def disable_update(self):
        self.update = False

    def update_min_max(self, name, t):
        new_avg = 0.0
        
        if self.update and name in self.stage.keys():
            if t < self.stage[name]['min']: self.stage[name]['min'] = t
            if t > self.stage[name]['max']: self.stage[name]['max'] = t
            
            new_avg = self.stage[name]['avg'] * 0.9 + t * 0.1
            self.stage[name]['avg'] = new_avg

    def summary(self):
        for name, t in self.stage.items():
           pass

def TimeLimit(x):
    max_time_end = time.time() + (60 * x)
    while True:
        print("TEST")
        if time.time() > max_time_end:
            break
