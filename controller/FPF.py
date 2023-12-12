import numpy as np

class FPF_controller:
    """
    This controller implements the control term in the feedback particle filter.
    Reference: Yang, Tao, Henk AP Blom, and Prashant G. Mehta. "The continuous-discrete time feedback particle filter." 2014 American Control Conference. IEEE, 2014.
    """
    __author__ = 'bryantzhou'

    def __init__(self, particle_num: int, sensor_gradient: np.ndarray, observation_dim: int, state_dim: int):
        assert isinstance(particle_num, int)
        assert particle_num > 0

        assert isinstance(observation_dim, int)
        assert observation_dim > 0

        assert isinstance(state_dim, int)
        assert state_dim > 0

        self.N = particle_num
        self.m = observation_dim
        self.d = state_dim
        self.grad_h = sensor_gradient

        self.h_hat = NotImplemented
        self.h_hat_sq = NotImplemented
        self.h_sq_hat = NotImplemented

    def observation_mean(self, observations: np.ndarray):
        self.h_hat = np.mean(observations,axis=1)
        self.h_hat_sq = self.h_hat.T @ self.h_hat

    def observation_square(self, observations: np.ndarray):
        h_sq = 0
        for m in range(self.N):
            h_sq += observations[:,m].T @ observations[:,m]
        self.h_sq_hat = 1/self.N * h_sq

    def control_K(self, prior_state: np.ndarray):
        K = np.zeros((self.d,self.m))
        self.observation_mean(prior_state)
        for m in range(self.N):
            K += prior_state[:,m].reshape((3,1)) @ (prior_state[:,m]-self.h_hat).reshape(1,3)
        K = 1/self.N * K 
        return K

    def control_sigma(self, prior_state: np.ndarray, control_K: np.ndarray):
        self.observation_mean(prior_state)
        self.observation_square(prior_state)

        g_hat = -(self.h_hat_sq - self.h_sq_hat)
        sigma = np.zeros(self.m)
        g = 0
        for p in range(self.m):
            g += control_K[:,p].T @ self.grad_h[:,p]
        for m in range(self.N):
            sigma += prior_state[:,m] * (g_hat - g)

        sigma = 1/self.N * sigma

        return sigma