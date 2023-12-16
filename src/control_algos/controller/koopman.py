import numpy as np
from control_algos.controller.LQR import dlqr, dlqr_finite


def koopman_control(X: np.ndarray, Y: np.ndarray, method: str, 
                    B: np.ndarray = np.zeros([2,2]), 
                    Q: np.ndarray = np.zeros([2,2]), 
                    R: np.ndarray = np.zeros([2,2]), steps: int = 0, 
                    Qf: np.ndarray = np.zeros([2,2])) -> np.ndarray:
    '''
    Learn liner dynamics from x_{k+1} = Ax_k, and then doing control with 
    the selected controller.

    Parameters:
    - X (matrix): A n by m matrix.  
    - Y (matrix): A n by m matrix. 
    - method (string): Specifing the controller. 
    
    - B (matrix): A n by r matrix, only needs to be provided if 
    using either dlqr or dlqr_finite.
    
    - Q (matrix): The n by n penalty matrix for states not being 0, only needs
    to be provided if using either dlqr or dlqr_finite.
    
    - R (matrix): The r by r penalty matrix for control effort (energy), only
    needs to be provided if using either dlqr or dlqr_finite.
    
    - Qf (matrix): The n by n penalty matrix for the state at final step N 
    not being 0, only needs to be provided if using dlqr_finite.
    
    - step (int): The total time steps, from 0 to N, only needs to
    be provided if using dlqr_finite.

    Returns:
    matrix: A with the dimension of n by n.
    or
    matrix: K with the dimension of r by n.
    '''
    A = Y @ np.linalg.pinv(X);
    
    if method == "PID":
        return A
    
    if method == "dlqr":
        K = dlqr(A, B, Q, R)
        return K
    
    if method == "dlqr_finite":
        K = dlqr_finite(A, B, Q, R, Qf, steps)
        return K
    
    raise Exception("Please specify the integrator: PID, dlqr, or dlqr_finite")
 