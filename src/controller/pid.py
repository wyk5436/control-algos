class pid_controller:
    """
    This class implements the classical proportional, integral, derivative controller
    """
    __author__ = 'bryantzhou'

    def __init__(self, k_p: float, k_i: float, k_d: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.u_p = 0.
        self.u_i = 0.
        self.u_d = 0.

        self.prev_error = 0.
        self.integral = 0.

    def control_p(self, error):
        return self.k_p * error

    def control_i(self, error):
        self.integral += error
        return self.k_i * self.integral

    def control_d(self, error):
        return self.k_d * (error - self.prev_error)

    def control_pid(self, error):
        control_total = self.control_p(error=error) + self.control_i(error=error) + self.control_d(error=error)
        self.prev_error = error
        return control_total