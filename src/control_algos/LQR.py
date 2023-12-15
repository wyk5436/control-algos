import numpy as np
import scipy

def controllability(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Calculate the controllability matrix, defined as [B A*B A^2*B ... A^{n-1}B]

    Parameters:
    - A (matrix): A n by n matrix.
    - B (matrix): A n by r matrix.

    Returns:
    matrix: R with the dimension of n by n*r.
    '''
    n = np.shape(A)[0]
    r = np.shape(B)[1]
    R = np.zeros([n,n*r])
    for i in range(0, n):
        R[:, i:i + 1] = np.linalg.matrix_power(A, i) @ B
    return R


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
         R: np.ndarray) -> np.ndarray:
    '''
    Find the optimal feedback control u = -Kx, with cost defined as J = 
    sum_{k = 0}^Inf (1/2)*x_k.T*Q*x_k + (1/2)*u_k.T*R*u_k. Note that the time
    horizion is infinity.

    Parameters:
    - A (matrix): A n by n matrix.
    - B (matrix): A n by r matrix.
    - Q (matrix): The n by n penalty matrix for states not being 0.
    - R (matrix): The r by r penalty matrix for control effort (energy).

    Returns:
    matrix: K with the dimension of r by n.
    '''
    n = np.shape(A)[0]
    romeo = controllability(A, B)
    
    if np.linalg.matrix_rank(romeo) < n:
        raise Exception("Can't use discrete LQR "
                        "because system is uncontrollable.")
    
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    
    return K


def dlqr_finite(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                Qf: np.ndarray, steps: int) -> np.ndarray:
    '''
    Find the optimal feedback control sequence u_i = -K_i x, with cost
    defined as J = sum_{k = 0}^N (1/2)*x_k.T*Q*x_k + (1/2)*u_k.T*R*u_k. Time
    horizion is infinity, so addition parameters are required.

    Parameters:
    - A (matrix): A n by n matrix.
    - B (matrix): A n by r matrix.
    - Q (matrix): The n by n penalty matrix for states not being 0.
    - R (matrix): The r by r penalty matrix for control effort (energy).
    - Qf (matrix): The n by n penalty matrix for
    the state at final step N not being 0. 
    - step (int): The total time steps, from 0 to N.

    Returns:
    3D matrix: Ks with the dimension of (step - 1) by r by n. The control 
    sequence is consist of time-varying K at time step 0 to N - 1.
    '''
    n = np.shape(A)[0]
    r = np.shape(B)[1]
    Ks = np.zeros([steps - 1, r, n])
    romeo = controllability(A, B)
    
    if np.linalg.matrix_rank(romeo) < n:
        raise Exception("Can't use discrete LQR "
                        "because system is uncontrollable.")
    
    P = Qf
    for i in range(steps - 2, -1, -1):
        P_new = (A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B)
                 @ B.T @ P @A + Q)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        Ks[i] = K
        P = P_new
        
    return Ks
