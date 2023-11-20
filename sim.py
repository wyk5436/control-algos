import numpy as np

def simulate_linear_disc(A, B, C, u, x0):
    '''
    Simulate discrete dynamics, with the state-space system having the form of
    x_{k+1} = A*x_k + B*u_k
    y_k = C*x_k

    Parameters:
    - A (matrix): A n by n matrix.
    - B (matrix): A n by m matrix.
    - C (matrix): A l by n matrix.
    - u (matrix): A n by k vector, representing controls at all k timesteps.
    - x0 (vector): A n by 1 vector, for the initial condition.
    

    Returns:
    matrix: y with the dimension of n by (k+1), including the initial condition
    '''
    n = np.shape(x0)[0]
    k = np.shape(u)[1]
    l = np.shape(C)[0]
    
    x = np.zeros([n, k + 1])
    y = np.zeros([l, k + 1])
    x[:, 0] = x0.reshape(-1);
    y[:, 0] = np.reshape(C @ x0, -1)
    for i in range(1, k + 1):
        x[:, i:i + 1] = A @ x[:, i - 1:i] + B @ u[:, i - 1:i]
        y[:, i:i + 1] = C @ x[:, i:i + 1]
        
    return y

        