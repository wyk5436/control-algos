import numpy as np
from control_algos.models.inver_pen import inverted_pen
from control_algos.controller.koopman import koopman_control
from control_algos.sim import simulate_linear_disc

pen = inverted_pen(1, 2, 0.8, 0.3);
B = np.array([[1],[0],[-1],[1]])
Q = np.diag(np.ones(4)*10)
R = np.array([[1]])


def generate_training_data(pendulum):
    theta = np.linspace(-0.2,0.2,30)
    thetad = np.linspace(-1,1,5);
    alphadot = np.linspace(-1,1,5);
    
    X = np.zeros([4, 30*5*5])
    Y = np.zeros([4, 30*5*5])
    
    count = 0
    for t in theta:
        for td in thetad:
            for ad in alphadot:
                x0 = np.array([[0],[t],[ad],[td]])
                xdot = pendulum.f(0, x0)
                X[:,count: count + 1] = x0
                Y[:,count: count + 1] = xdot*0.1 + x0;
                count = count + 1    
    return X, Y

[X,Y] = generate_training_data(pen)
K = koopman_control(X, Y, "dlqr", B, Q, R)

x0 = np.array([[0],[0.1],[0],[0]])
trajectory = simulate_linear_disc(Y@np.linalg.pinv(X), B, np.eye(4), 
                                  np.zeros([4,1]), x0, K, 100)
theta = trajectory[1,:]


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 9), dpi=80)

plt.plot(np.linspace(0,10,101),np.rad2deg(theta))

plt.xlabel('time')  
plt.ylabel('theta')  

  
plt.title("Balancing an inverted pendulum") 
plt.savefig('balanced.png')
plt.show()