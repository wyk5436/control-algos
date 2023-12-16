import numpy as np
import numbers
from control_algos.integrator.stochastic_integrator import stochastic_integrator


class WienerRK4Maruyama(stochastic_integrator):
    """
    WienerRK4Maruyama class documentation string

            It solves Wiener stochastic differential equations of the form:

            dX_t = a(t, X_t) dt + dW_t

            Where:
                a(t, X_t) is the drift_vector_field

            It uses an RK4 (deterministic) Scheme for a(t, X_t) dt

    """

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
            if self.deterministic:
                dW = 0.
            else:
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
    
    def get_noise(self):
        return self.noise