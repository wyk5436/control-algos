class L63:
    """The Lorenz 1963 Model."""

    __author__ = 'bryantzhou'

    def __init__(self, sigma=None, rho=None, beta=None, parameters=None):
        super().__init__()

        assert isinstance(parameters, (type(None), dict))

        if parameters is None:
            assert isinstance(sigma, float)
            assert isinstance(rho, float)
            assert isinstance(beta, float)
        else:
            items = ['sigma', 'rho', 'beta']
            for item in items:
                assert item in parameters.keys()
                assert isinstance(parameters[item], float)
                assert parameters[item] > 0.0
            sigma = parameters['sigma']
            rho = parameters['rho']
            beta = parameters['beta']

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    @staticmethod
    def get_standard_parameters(set_name='Lorenz 1963'):
        assert set_name in ['Lorenz 1963']

        parameter_set = {'sigma': 0.0, 'rho': 0.0, 'beta': 0.0}

        if set_name == 'Lorenz 1963':
            parameter_set['sigma'] = 10.0
            parameter_set['rho'] = 28.0
            parameter_set['beta'] = 8.0 / 3.0

        return parameter_set

    def evaluate(self, stepsize, time, states):

        ds = states.copy()
        ds[0] = self.sigma * (states[1] - states[0])
        ds[1] = self.rho * states[0] - states[1] - states[0]*states[2]
        ds[2] = states[0] * states[1] - self.beta * states[2]

        return ds
