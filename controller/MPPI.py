import numpy as np

from models.UAV import UAV_model
from integrator.wiener_rk4_maruyama import WienerRK4Maruyama

class MPPI_controller:
    """
    This class implements the model predictive path integral controller
    """
    __author__ = "bryantzhou"

    def __init__(self, MC_run_total: int,
                 current_time: float, 
                 final_time: float, 
                 time_step: float, 
                 current_states: np.ndarray,
                 k: float,
                 eta: float,
                 d: float,
                 l: float,
                 a: float,
                 g2: float,
                 xR1: float,
                 xR2: float,
                 xS1: float,
                 xS2: float,
                 yR1: float,
                 yR2: float,
                 yS1: float,
                 yS2: float,
                 xP: float,
                 xQ: float,
                 yP: float,
                 yQ: float):
        self.MC_run_total = MC_run_total
        self.t = current_time
        self.T = final_time
        self.dt = time_step
        self.x = current_states

        self.safety_flag = True

        self.eta = eta
        self.d = d
        self.l = l
        self.a = a
        self.g2 = g2

        self.xR1 = xR1
        self.xR2 = xR2
        self.xS1 = xS1
        self.xS2 = xS2
        self.yR1 = yR1
        self.yR2 = yR2
        self.yS1 = yS1
        self.yS2 = yS2

        self.xP = xP
        self.xQ = xQ
        self.yP = yP
        self.yQ = yQ
        
        self.EOM = UAV_model(k=k)
        self.integrator = WienerRK4Maruyama(stepsize=time_step)
        self.integrator.add_drift_vector_field(drifting=self.EOM.evaluate)
        self.integrator.set_brownian_motion_parameters(covariance_matrix=0.01*np.identity(2, float), number_of_zeros_to_append=2)
        self.integrator.never_clear_history(never_clear_history=False)

    def check_safety(self, current_states):
        if ((current_states[0] >= self.xR1 and current_states[0] <= self.xS1 and current_states[1] >= self.yR1 and current_states[1] <= self.yS1) or 
            (current_states[0] >= self.xR2 and current_states[0] <= self.xS2 and current_states[1] >= self.yR2 and current_states[1] <= self.yS2) or
            (current_states[0] <= self.xP or current_states[0] >= self.xQ or current_states[1] <= self.yP or current_states[1] >= self.yQ)):
            self.safety_flag = False
            return False
        else:
            return True

    def running_cost(self, current_states:np.ndarray):
        return (current_states[0]**2 + current_states[1]**2)

    def total_cost(self):
        total_cost = 0
        self.safety_flag = True
        # self.integrator.set_fixed_seed(seed=seed)

        current_states = self.x.copy()
        for count, t in enumerate(np.linspace(self.t, self.T, int((self.T-self.dt-self.t)/self.dt)+1)):
            # calculate the running cost
            running_cost = self.running_cost(current_states=current_states.copy())
            total_cost += running_cost*self.dt

            self.integrator.evaluate(s=current_states, t0=t, tf=t+self.dt)
            current_states = self.integrator.get_states().T[:,-1]
            if count == 0:
                noise = self.integrator.get_noise().T[:,-1]

            self.check_safety(current_states=current_states.copy())

            if self.safety_flag == False:
                total_cost += self.eta
                break

        if self.safety_flag == True:
            total_cost += self.d * (current_states[0]**2 + current_states[1]**2)

        return total_cost, noise
    
    def controal_law(self, cost, noise):
        denom_i = np.exp(-cost/self.l)
        numer = noise @ denom_i.reshape((self.MC_run_total,1))
        denom = np.sum(denom_i)

        ut = np.identity(2) * (1-self.a/self.g2)**(-1) @ numer/(np.sqrt(self.dt)*denom)
        vt = -np.identity(2) * (self.g2/self.a-1)**(-1) @ numer/(np.sqrt(self.dt)*denom)

        return ut, vt

    
    def propogate_with_control(self, current_states: np.ndarray, start_time: float, finish_time: float, u, v):
        self.EOM.get_control_parameters(u=u, v=v)

        Integrator = WienerRK4Maruyama(stepsize=self.dt)
        Integrator.add_drift_vector_field(drifting=self.EOM.evaluate_with_control)
        # Integrator.set_brownian_motion_parameters(covariance_matrix=0.01*np.identity(2, float), number_of_zeros_to_append=2)
        Integrator.never_clear_history(never_clear_history=False)
        # Integrator.set_random_seed(seed=seed)
        Integrator.set_deterministic()

        Integrator.evaluate(s=current_states,t0=start_time, tf=finish_time)

        return Integrator.get_states().T[:,-1]