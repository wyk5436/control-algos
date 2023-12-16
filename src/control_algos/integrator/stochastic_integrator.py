import numbers
import numpy as np
import copy


class stochastic_integrator:

    def __init__(self, stepsize: (numbers.Real, type(None)) = None, number_of_steps: (int, type(None)) = None):
        # initialize the drift term
        self.drift_vector_field = []
        self.clear_drift_vector_field()

        # set the desired time step if given
        self._stepsize = None
        self.set_stepsize(stepsize=stepsize)

        # set the number of steps if given
        self._number_of_steps = None
        self.set_number_of_steps(number_of_steps=number_of_steps)
        self.default_number_of_steps = 1000

        # initialize states
        self.states = None

        # set integrator tolerance
        self.tolerance = 1E-9
        
        # initialize parameters for noise terms
        self._square_root_of_stepsize = NotImplemented
        self.brownian_motion_dimension = NotImplemented
        self.number_of_zeros_to_append = NotImplemented

        # initialize history parameters
        self._save_integration_history = True
        self._never_clear_history = False

        self._user_prefers_stepsize_versus_number_of_steps = True

        self.apply_random_seed = False
        self.random_seed = NotImplemented
        self.get_and_set_state_of_random_number_generator = False
        self.random_number_generator_state = NotImplemented
        self.fixed_seed_bool = False
        self.fixed_seed = NotImplemented

        self.deterministic = False

    # make sure the drift vector field is empty when instantiating a new integrator
    def clear_drift_vector_field(self):
        self.drift_vector_field = []

    # append each drifting term
    def add_drift_vector_field(self, drifting: callable):
        assert callable(drifting)
        self.drift_vector_field.append(drifting)

    def set_stepsize(self, stepsize: (type(None), int, float), update_user_preference: bool = True):
        assert isinstance(stepsize, (type(None), int, float))
        assert isinstance(update_user_preference, bool)

        if isinstance(stepsize, int):
            stepsize = float(stepsize)

        if isinstance(stepsize, float):
            assert stepsize > 0.0
            if update_user_preference:
                self._user_prefers_stepsize_versus_number_of_steps = True

        self._stepsize = stepsize

    def set_number_of_steps(self, number_of_steps: (int, type(None)), update_user_preference: bool = True):
        assert isinstance(number_of_steps, (type(None), int))
        assert isinstance(update_user_preference, bool)

        if isinstance(number_of_steps, int):
            assert number_of_steps > 0
            if update_user_preference:
                self._user_prefers_stepsize_versus_number_of_steps = False

        self._number_of_steps = number_of_steps

    # calculate the square root of step size for random sampling purposes
    def _set_square_root_of_stepsize(self, square_root_of_stepsize):
        assert isinstance(square_root_of_stepsize, float)
        assert square_root_of_stepsize >= 0.0
        self._square_root_of_stepsize = square_root_of_stepsize

    def never_clear_history(self, never_clear_history: bool = False):
        assert isinstance(never_clear_history, bool)
        self._never_clear_history = never_clear_history

    def set_fixed_seed(self, seed: int):
        self.fixed_seed = seed
        self.fixed_seed_bool = True

    def set_deterministic(self):
        self.deterministic = True
    
    # generate random samples
    def generate_brownian_increment_with_covariance(self) -> np.ndarray:
        if self.fixed_seed_bool:
            np.random.seed(self.fixed_seed)
        noise = self.brownian_motion_cholesky.dot(np.random.normal(loc=0.,
                                                         scale=self.brownian_motion_standard_deviation
                                                               * self._square_root_of_stepsize,
                                                               size=self.brownian_motion_dimension))
        noise += self.brownian_motion_mean
        return noise
    
    def generate_brownian_increment_with_covariance_and_append_zeros(self) -> np.ndarray:
        noise = np.zeros((self.brownian_motion_dimension + self.number_of_zeros_to_append))
        noise[self.number_of_zeros_to_append:] = self.generate_brownian_increment_with_covariance()
        return noise
    
    # set up brownian motion parameters
    def set_brownian_motion_parameters(self,
                                       covariance_matrix: [np.ndarray, None] = None,
                                       mean: [float, np.ndarray] = 0.0,
                                       variance: float = 1.0,
                                       number_of_zeros_to_append: [None, int] = None):
        self.brownian_motion_mean = mean
        self.brownian_motion_covariance_matrix = covariance_matrix
        self.brownian_motion_standard_deviation = 1.
        self.number_of_zeros_to_append = number_of_zeros_to_append
        self.brownian_motion_cholesky = np.linalg.cholesky(covariance_matrix)
        if number_of_zeros_to_append is None:
            self.generate_brownian_increment = self.generate_brownian_increment_with_covariance
        else:
            self.generate_brownian_increment = self.generate_brownian_increment_with_covariance_and_append_zeros
        self.brownian_motion_dimension = self.brownian_motion_cholesky.shape[1]


    def deepcopy(self):
        return copy.deepcopy(self)
    
    def setup(self, t0: numbers.Real, tf: numbers.Real):
        if self._stepsize is None and self._number_of_steps is None:
            # if neither stepsize nor number_of_steps has been set, then number_of_steps will get the preference.
            self.set_number_of_steps(number_of_steps=self.default_number_of_steps)
            self.set_stepsize(stepsize=abs(tf - t0) / self._number_of_steps, update_user_preference=False)
        elif self._stepsize is None or not self._user_prefers_stepsize_versus_number_of_steps:
            # stepsize has not been set, but number of steps has. again, preference will be for number of steps.
            self.set_stepsize(stepsize=abs(tf - t0) / self._number_of_steps, update_user_preference=False)
        elif self._number_of_steps is None or self._user_prefers_stepsize_versus_number_of_steps:
            #  number_of_steps was not set, but stepsize was. preference will be for stepsize.
            number_of_steps = abs(int((tf - t0) / self._stepsize))
            #  check to see if the number of steps is sufficient, we may actually need one more in fringe cases
            if abs((tf - t0) - number_of_steps * self._stepsize) > self.tolerance:
                number_of_steps += 1
            self.set_number_of_steps(number_of_steps=number_of_steps, update_user_preference=False)
        else:
            raise UserWarning('some logic in the class is not complete.')
    
    # set up the sorage for history of states, time, and noise
    def _setup_storage(self,
                       state: np.ndarray,
                       t0: numbers.Real,
                       dim: int,
                       clear_history: bool,
                       tol: numbers.Real = 1E-12) -> (int, None):

        if self._save_integration_history:
            if isinstance(self.states, np.ndarray) and (not clear_history or self._never_clear_history):
                assert hasattr(self, 'time')
                assert hasattr(self, 'noise')

                index_to_return = len(self.states)

                if np.allclose(self.states[-1], state, atol=tol) and abs(self.time[-1] - t0) < 1E-12:
                    #  states
                    self.states = np.vstack((self.states, np.zeros([self._number_of_steps, dim])))
                    self.noise = np.vstack((self.noise, np.zeros([self._number_of_steps, dim])))
                    #  time
                    self.time = np.hstack((self.time, np.zeros(self._number_of_steps, )))
                else:
                    #  states
                    self.states = np.vstack((self.states, np.zeros([self._number_of_steps + 1, dim])))
                    self.states[-(self._number_of_steps + 1), :] = state
                    self.noise = np.vstack((self.noise, np.zeros([self._number_of_steps + 1, dim])))
                    index_to_return += 1
                    #  time
                    self.time = np.hstack((self.time, np.zeros(self._number_of_steps + 1, )))
                    self.time[-(self._number_of_steps + 1)] = t0
            else:
                index_to_return = 1
                self.states = np.zeros([self._number_of_steps + 1, dim])
                self.noise = self.states.copy()
                self.states[0] = state
                self.time = np.zeros(self._number_of_steps + 1)
                self.time[0] = t0

            self._storage_starting_index = index_to_return

            return index_to_return
    