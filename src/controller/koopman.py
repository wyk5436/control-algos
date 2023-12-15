import numpy as np
from src.controller.LQR import dlqr, dlqr_finite


def koopman_control(X, Y, method, B = 0, Q = 0, R = 0, steps = 0, Qf = 0):
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
 