from eom.eom import EOM
import math

class bicycle_model(EOM):
    """The bicycle model"""

    __author__ = "bryantzhou"

    def __init__(self, L=None, delta=None, parameters=None):

        assert isinstance(parameters, (type(None), dict))

        if parameters is None:
            assert isinstance(L, float)
            assert isinstance(delta, float)
        else:
            items = ['L', 'delta']
            for item in items:
                assert item in parameters.keys()
                assert isinstance(parameters[item], float)
            L = parameters['L']
            delta = parameters['delta']

        self.L = L
        self.delta = delta

    @staticmethod
    def get_standard_parameters():

        parameter_set = {'L': 0.0, 'delta': 0.0}

        parameter_set['L'] = 5.0
        parameter_set['delta'] = 10.0

        return parameter_set
    
    def _evaluate(self, stepsize, time, states):

        ds = states.copy()

        ds[0] = states[3] * math.cos(states[2])
        ds[1] = states[3] * math.sin(states[2])
        ds[2] = states[3] * math.tan(self.delta) / self.L
        ds[3] = 0

        return ds