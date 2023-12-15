from eom.eom import EOM
import math

class bicycle_model(EOM):
    """The bicycle model"""

    __author__ = "bryantzhou"

    """
    states:[x,
            y,
            theta,
            v]
    They are: x, y: vehicle's 2D position; 
                theta: the angle of the bicycle’s forwards direction with respect to the x-axis;
                v: vehicle's velocity
    
    Control inputs: a: acceleration; delta: steering angle
    """

    def __init__(self, L, a, delta):

        self.L = L
        self.a = a
        self.delta = delta
    
    def _evaluate(self, stepsize, time, states):

        ds = states.copy()

        ds[0] = states[3] * math.cos(states[2])
        ds[1] = states[3] * math.sin(states[2])
        ds[2] = states[3] * math.tan(self.delta) / self.L
        ds[3] = self.a

        return ds