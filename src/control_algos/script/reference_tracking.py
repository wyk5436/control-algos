import numpy as np
from matplotlib.pyplot import *

from control_algos.controller.pid import pid_controller
from control_algos.models.UAV import UAV_model
from control_algos.integrator.wiener_rk4_maruyama import WienerRK4Maruyama

# define parameters
t = 0.
T = 10.
dt = 0.01
x0 = np.array((50., 0., 0.5, np.pi/2))
x = np.zeros((4,int(T/dt)+1))
control_input = np.zeros(2)

k_p = 0.01
k_i = 0.
k_d = 0.05

# Create reference speed
v_ref = np.ones(int(T/dt)+1)
v_ref[:500] = 5.
v_ref[500:] = 10.

radius_ref = 50.0

controller = pid_controller(k_p=k_p, k_i=k_i, k_d=k_d)
EOM = UAV_model(k=0)
Integrator = WienerRK4Maruyama(stepsize=dt)

xt = x0
x[:,0] = x0
for count, t in enumerate(np.linspace(t, T-dt, int((T-dt)/dt+1))):
    EOM.get_control_parameters(u=control_input[0], v=control_input[1])

    Integrator.add_drift_vector_field(drifting=EOM.evaluate_with_control_pid)
    Integrator.never_clear_history(never_clear_history=False)
    Integrator.set_deterministic()

    Integrator.evaluate(s=xt.copy(),t0=t, tf=t+dt)
    xt = Integrator.get_states().T[:,-1]

    # error = v_ref[count] - xt[2]
    error = radius_ref - np.sqrt(xt[0]**2 + xt[1]**2)

    control_input[1] = controller.control_pid(error=error)

    x[:,count+1] = xt

# figure(1)
# plot(np.linspace(0.0, T, int(T/dt+1)), v_ref, 'k-', label="reference")
# plot(np.linspace(0.0, T, int(T/dt+1)), x[2,:], 'b-', label="UAV")
# ylim([0,15])
# legend()
# show()

# Create a theta array for angles from 0 to 2*pi
theta = np.linspace(0, 2*np.pi, 100)

# Circle parameters
center = (0, 0)

# Calculate the x and y coordinates of the circle
x_circle = center[0] + radius_ref * np.cos(theta)
y_circle = center[1] + radius_ref * np.sin(theta)

figure(2)
plot(x_circle, y_circle, label='Circle')
plot(x[0,:], x[1,:], 'b-', label="UAV")
legend()
show()


