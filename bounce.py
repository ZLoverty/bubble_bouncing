"""
bounce.py
=========

Description
-----------

Rewrite the bubble simulation in object oriented way. We aim to improve the modularity and readability of the simulator.

"""
import numpy as np
from scipy import integrate

class BounceSimulator:
    def __init__(self):
        self.print_interval = 0.1
        self.last_print = 0.0
        pass
    

      
    def run(self):

        def film_drainage(t, y):
            return 2*y
        
        def event_print(t, y):
            print(self.print_interval)
            if t - self.last_print > self.print_interval:
                print(t)
                self.last_print = t 

        sol = integrate.solve_ivp(film_drainage, [0, 1], np.array([1]), method="BDF", events=event_print)

sim = BounceSimulator()
sim.run()
