import numpy as np

class inverted_pen:
    
    '''
    Nonlinear inverted pendulumn sitting on a rotary arm.
    '''
    
    def __init__(self, m: float, M: float, l: float, r: float):
        self.m = m
        self.M = M
        self.l = l
        self.r = r
        self.g = 9.81 # Assuming we are on earth...
        
    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        '''
        Calculates the nonlinear f(x) for xdot = f(x)

        Parameters:
        - t (float): time.
        - x (vector): states.
        
        Returns:
        vector: f(x).
        '''
        x = x.flatten()
        alpha = x[0]
        theta = x[1]
        alphad = x[2]
        thetad = x[3]
    
        alpha_dot = x[2]
        theta_dot = x[3]
    
        denom = ((self.M / 4 + self.m)*self.r**2 + self.m * self.l**2 * 
                 np.sin(theta)**2 - self.m * self.r**2 * np.cos(theta)**2)
        
        alpha_ddot = ((self.m*self.r*self.l*np.cos(theta)*(alphad**2 * 
                      np.sin(theta) * np.cos(theta) + self.g*np.sin(theta) / 
                      self.l) - self.m*self.r*self.l*thetad**2*np.sin(theta) - 
                      2*self.m*self.l**2 * alphad*thetad * 
                      np.sin(theta)*np.cos(theta)) / denom)
    
        K = (self.M/4 + self.m)*self.r**2 + self.m*self.l**2*np.sin(theta)**2
        
        thetaddot_numer = ((self.r/self.l) * (-self.m*self.r*self.l * 
                          thetad**2*np.sin(theta) - 2*self.m*self.l**2 * 
                          alphad*thetad*np.sin(theta)*np.cos(theta)) * 
                          np.cos(theta) / K + alphad**2*np.sin(theta) * 
                          np.cos(theta) + self.g*np.sin(theta) / self.l)

        thetaddot_denom = 1 - self.m*self.r**2 * np.cos(theta)**2 / K
        theta_ddot = thetaddot_numer / thetaddot_denom
        
        return np.array([[alpha_dot], [theta_dot], [alpha_ddot], [theta_ddot]])