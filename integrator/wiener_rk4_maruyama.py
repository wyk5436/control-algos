import numpy as np
import numbers
import copy


class WienerRK4Maruyama:
    """
    WienerRK4Maruyama class documentation string

            It solves Wiener stochastic differential equations of the form:

            dX_t = a(t, X_t) dt + dW_t

            Where:
                a(t, X_t) is the drift_vector_field

            It uses an RK4 (deterministic) Scheme for a(t, X_t) dt

    """

    __author__ = 'rynebeeson'

    def __init__(self, stepsize: (numbers.Real, type(None)) = None, number_of_steps: (int, type(None)) = None):
        self.drift_vector_field = []
        self.clear_drift_vector_field()

        # set the desired time step if given
        self._stepsize = None
        self.set_stepsize(stepsize=stepsize)

        self._number_of_steps = None
        self.set_number_of_steps(number_of_steps=number_of_steps)

        self.tolerance = 1E-9
        self.default_number_of_steps = 1000

        self._square_root_of_stepsize = NotImplemented
        self.brownian_motion_dimension = NotImplemented

        self._user_prefers_stepsize_versus_number_of_steps = True

        self._save_integration_history = True

        self.states = None

        self._never_clear_history = False
    
    def never_clear_history(self, never_clear_history: bool = False):
        assert isinstance(never_clear_history, bool)
        self._never_clear_history = never_clear_history

    def clear_drift_vector_field(self):
        self.drift_vector_field = []

    def add_drift_vector_field(self, drifting: callable):
        assert callable(drifting)
        self.drift_vector_field.append(drifting)

    def set_stepsize(self, stepsize: (type(None), int, float)):
        assert isinstance(stepsize, (type(None), int, float))

        if isinstance(stepsize, int):
            stepsize = float(stepsize)

        if isinstance(stepsize, float):
            assert stepsize > 0.0

        self._stepsize = stepsize

    def set_number_of_steps(self, number_of_steps: (int, type(None)), update_user_preference: bool = True):
        assert isinstance(number_of_steps, (type(None), int))
        assert isinstance(update_user_preference, bool)

        if isinstance(number_of_steps, int):
            assert number_of_steps > 0
            if update_user_preference:
                self._user_prefers_stepsize_versus_number_of_steps = False

        self._number_of_steps = number_of_steps

    def _set_square_root_of_stepsize(self, square_root_of_stepsize):
        assert isinstance(square_root_of_stepsize, float)
        assert square_root_of_stepsize >= 0.0
        self._square_root_of_stepsize = square_root_of_stepsize
    
    def _generate_brownian_increment_with_covariance(self) -> np.ndarray:
        noise = self.brownian_motion_cholesky.dot(np.random.normal(loc=0.,
                                                         scale=self.brownian_motion_standard_deviation
                                                               * self._square_root_of_stepsize,
                                                               size=self.brownian_motion_dimension))
        noise += self.brownian_motion_mean
        return noise
    
    def set_brownian_motion_parameters(self,
                                       covariance_matrix: [np.ndarray, None] = None,
                                       mean: [float, np.ndarray] = 0.0,
                                       variance: float = 1.0):
        self.brownian_motion_mean = mean
        self.brownian_motion_covariance_matrix = covariance_matrix
        self.brownian_motion_standard_deviation = 1.
        self.brownian_motion_cholesky = np.linalg.cholesky(covariance_matrix)
        self.generate_brownian_increment = self._generate_brownian_increment_with_covariance
        self.brownian_motion_dimension = self.brownian_motion_cholesky.shape[1]

    def deepcopy(self):
        return copy.deepcopy(self)
    
    def setup(self, t0: numbers.Real, tf: numbers.Real):
        # neither stepsize nor number_of_steps has been set. number_of_steps will get the preference.
        if self._stepsize is None and self._number_of_steps is None:
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
    
    def _setup_storage(self,
                       state: np.ndarray,
                       t0: numbers.Real,
                       dim: int,
                       clear_history: bool,
                       tol: numbers.Real = 1E-12) -> (int, None):

        if self._save_integration_history:
            from numpy import zeros
            if isinstance(self.states, np.ndarray) and (not clear_history or self._never_clear_history):
                assert hasattr(self, 'time')
                assert hasattr(self, 'noise')
                from numpy import vstack, hstack, allclose

                index_to_return = len(self.states)

                if allclose(self.states[-1], state, atol=tol) and abs(self.time[-1] - t0) < 1E-12:
                    #  states
                    self.states = vstack((self.states, zeros([self._number_of_steps, dim])))
                    self.noise = vstack((self.noise, zeros([self._number_of_steps, dim])))
                    #  time
                    self.time = hstack((self.time, zeros(self._number_of_steps, )))
                else:
                    #  states
                    self.states = vstack((self.states, zeros([self._number_of_steps + 1, dim])))
                    self.states[-(self._number_of_steps + 1), :] = state
                    self.noise = vstack((self.noise, zeros([self._number_of_steps + 1, dim])))
                    index_to_return += 1
                    #  time
                    self.time = hstack((self.time, zeros(self._number_of_steps + 1, )))
                    self.time[-(self._number_of_steps + 1)] = t0
            else:
                index_to_return = 1
                self.states = zeros([self._number_of_steps + 1, dim])
                self.noise = self.states.copy()
                self.states[0] = state
                self.time = zeros(self._number_of_steps + 1)
                self.time[0] = t0

            self._storage_starting_index = index_to_return

            return index_to_return
        
    def setup(self, t0: numbers.Real, tf: numbers.Real):
        # neither stepsize nor number_of_steps has been set. number_of_steps will get the preference.
        if self._stepsize is None and self._number_of_steps is None:
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

    def evaluate(self, s: np.ndarray, t0: numbers.Real, tf: numbers.Real, clear_history: bool = True):
        assert isinstance(s, np.ndarray)
        assert s.ndim == 1
        assert isinstance(t0, (int, float))
        assert isinstance(tf, (int, float))

        if isinstance(t0, int):
            t0 = float(t0)

        if isinstance(tf, int):
            tf = float(tf)

        #  no need to run the integrator if the tolerance is already achieved
        if abs(tf - t0) <= self.tolerance:
            from numpy import zeros_like
            # save the final states if we haven't been saving states along the way
            if not self._save_integration_history:
                self.states = s
                self.time = t0
                self.noise = zeros_like(s)
            else:
                self.states = np.array([s, s])
                self.time = np.array([t0, t0])
                self.noise = np.array([zeros_like(s), zeros_like(s)])
            return

        self.setup(t0, tf)

        #  finish the setup of the brownian noise generation
        self._set_square_root_of_stepsize(square_root_of_stepsize=np.sqrt(self._stepsize))

        #  create a 2d array for storing the states and noise and 1d for time
        index_for_solution_history = self._setup_storage(state=s,
                                                         t0=t0,
                                                         dim=s.size,
                                                         clear_history=clear_history,
                                                         tol=1E-12)

        #  set the initial time
        t = t0
        h = self._stepsize
        if (tf - t0) < 0.0:
            h *= -1.0
        h2 = h / 2.

        while abs(tf - t) > self.tolerance:

            if abs(h) > abs(tf - t):
                h = (tf - t)
                h2 = h / 2.
                self._set_square_root_of_stepsize(square_root_of_stepsize=np.sqrt(abs(h)))

            #  k1
            k1 = self.drift_vector_field[0](0.0, t, s)
            for vector_field in self.drift_vector_field[1:]: k1 += vector_field(0.0, t, s)
            sp = s + h2 * k1

            #  generate the brownian noise
            dW = self.generate_brownian_increment()

            #  k2
            t += h2
            k2 = self.drift_vector_field[0](h2, t, sp)
            for vector_field in self.drift_vector_field[1:]: k2 += vector_field(h2, t, sp)
            sp = s + h2 * k2

            #  k3
            k3 = self.drift_vector_field[0](h2, t, sp)
            for vector_field in self.drift_vector_field[1:]: k3 += vector_field(h2, t, sp)
            sp = s + (h * k3)

            #  k4
            t += h2
            k4 = self.drift_vector_field[0](h, t, sp)
            for vector_field in self.drift_vector_field[1:]: k4 += vector_field(h, t, sp)

            #  new state
            s += h*(k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 + dW

            #  save the current state and time
            if self._save_integration_history:

                try:
                    self.states[index_for_solution_history] = s
                    self.time[index_for_solution_history] = t
                    self.noise[index_for_solution_history] = dW
                except IndexError:
                    if abs(tf - t) < self._stepsize:
                        from numpy import concatenate, array

                        if self._flag_once_container_allocation_warning:
                            self._flag_once_container_allocation_warning = False

                            from warnings import warn
                            warn('In ' + self.__name__ + ' the a priori storage allocation was not enough '
                                                         'for saving the integrated solution. '
                                                         'Using numpy.concatenate')

                        self.states = concatenate((self.states, s.reshape((1, -1))), axis=0)
                        self.time = concatenate((self.time, array([t])))
                        self.noise = concatenate((self.noise, dW.reshape((1, -1))), axis=0)

                index_for_solution_history += 1

        # save the final states if we haven't been saving states along the way
        if not self._save_integration_history:
            self.states = s
            self.time = t
            self.noise = dW

    def get_states(self):
        return self.states
    
    def get_times(self):
        return self.time