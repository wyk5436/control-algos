import numpy as np
from typing import Callable

def simulate_linear_disc(A: np.ndarray, B: np.ndarray, 
                         C: np.ndarray, u: np.ndarray, x0: np.ndarray, 
                         K: np.ndarray = np.zeros([1,4]), 
                         steps = 0) -> np.ndarray:
    '''
    Simulate linear discrete dynamics, with the state-space system having the form of
    x_{k+1} = A*x_k + B*u_k
    y_k = C*x_k

    Parameters:
    - A (matrix): A n by n matrix.
    - B (matrix): A n by m matrix.
    - C (matrix): A l by n matrix.
    - u (matrix): A m by k vector, representing controls at all k timesteps.
    - x0 (vector): A n by 1 vector, for the initial condition.
    

    Returns:
    matrix: y with the dimension of l by (k+1), including the initial condition.
    '''
    n = np.shape(x0)[0]
    k = np.shape(u)[1]
    l = np.shape(C)[0]
    
    x = np.zeros([n, k + 1])
    y = np.zeros([l, k + 1])
    x[:, 0] = x0.reshape(-1);
    y[:, 0] = np.reshape(C @ x0, -1)
    if np.linalg.norm(K) != 0:
        x = np.zeros([n, steps + 1])
        y = np.zeros([l, steps + 1])
        x[:, 0] = x0.reshape(-1);
        y[:, 0] = np.reshape(C @ x0, -1)
        for i in range(1, steps + 1):
            x[:, i:i + 1] = A @ x[:, i - 1:i] + B @ (-K @ x[:, i - 1:i])
            y[:, i:i + 1] = C @ x[:, i:i + 1]     
    else:      
        for i in range(1, k + 1):
            x[:, i:i + 1] = A @ x[:, i - 1:i] + B @ u[:, i - 1:i]
            y[:, i:i + 1] = C @ x[:, i:i + 1]     
    return y

def simulate_nonlinear(f: Callable[[np.ndarray], np.ndarray], 
                       u: Callable[[int, np.ndarray], np.ndarray], C: np.ndarray, 
                       x0: np.ndarray, steps:int ) -> np.ndarray:
    '''
    Simulate nonlinear discrete dynamics, with the state-space system having the form of
    x_{k+1} = f(x_k) + u(k, x_k)
    y_k = C*x_k

    Parameters:
    - f (function): A function which outputs a n by 1 vector.
    - u (function): A function that outputs a n by 1 vector.
    - C (matrix): A l by n matrix.
    - x0 (vector): A n by 1 vector, for the initial condition.
    - steps (int): The integer that denotes the time step
    

    Returns:
    matrix: y with the dimension of l by (steps + 1), including the initial condition.
    '''
    n = np.shape(x0)[0]
    l = np.shape(C)[0]
    
    x = np.zeros([n, steps + 1])
    y = np.zeros([l, steps + 1])
    x[:, 0] = x0.reshape(-1);
    y[:, 0] = np.reshape(C @ x0, -1)
    for i in range(1, steps + 1):
        x[:, i:i + 1] = f(x[:, i - 1:i]) + u(i-1, x[:, i - 1:i])
        y[:, i:i + 1] = C @ x[:, i:i + 1]
    return y
