from eom.eom import EOM
import math

class UAV_model:
    """The UAV model, which can be considered as a point vehicle model."""

    __author__ = "bryantzhou"

    """
    states:[x,
            y,
            v,
            phi]
    They are:   v: vehicle's velocity;
                phi: vehicle's heading angle;
                x, y: vehicle's 2D position;
    
    Control inputs: a: acceleration; w: angular speed
    """

    def __init__(self, k: float):

        self.k = k

    
    def evaluate(self, stepsize, time, states):

        ds = states.copy()

        ds[0] = -self.k*states[0] + states[2] * math.cos(states[3])
        ds[1] = -self.k*states[1] + states[2] * math.sin(states[3])
        ds[2] = -self.k*states[2]
        ds[3] = -self.k*states[3]

        return ds
    
    def get_control_parameters(self, u, v):
        self.u = u
        self.v = v
    
    def evaluate_with_control(self, stepsize, time, states):

        ds = states.copy()

        ds[0] = -self.k*states[0] + states[2] * math.cos(states[3])
        ds[1] = -self.k*states[1] + states[2] * math.sin(states[3])
        ds[2] = -self.k*states[2] + self.u[0] + self.v[0]
        ds[3] = -self.k*states[3] + self.u[1] + self.v[1]

        return ds