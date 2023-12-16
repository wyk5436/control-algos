#  import and set the random number generator
import numpy as np
from matplotlib.pyplot import *

from control_algos.models.Lorenz_63 import L63
from control_algos.integrator.wiener_rk4_maruyama import WienerRK4Maruyama
from control_algos.controller.FPF import FPF_controller

np.random.seed(1)
#  parameters for continuous-discrete FPF
N = 500
d_lambda = 0.01
state_ini_mean = np.array([-3.13943612, -2.06681508, 22.90365922]) # initial mean state
t = 0.0
T = 10.0
dt = 0.01
obs_incre = 0.4 # obs_incre = 0.5 when testing for d_lambda
grad_h = np.eye(3)
state_all_mean = np.zeros((3,int(T/dt)+1))
state_all_x = np.zeros((int(T/dt)+1,N))
state_all_y = np.zeros((int(T/dt)+1,N))
state_all_z = np.zeros((int(T/dt)+1,N))
RMSE_t_x = np.zeros(int(T/dt)+1)
RMSE_t_y = np.zeros(int(T/dt)+1)
RMSE_t_z = np.zeros(int(T/dt)+1)

#  import and setup the L63 model
EOM = L63(parameters=L63.get_standard_parameters())

#  import an integrator and test L96
Integrator = WienerRK4Maruyama(stepsize=dt)
Integrator.add_drift_vector_field(drifting=EOM.evaluate)
Integrator.set_brownian_motion_parameters(covariance_matrix=1 * np.identity(3, float))

sIntegrator = Integrator.deepcopy()

state_ini_part = np.random.multivariate_normal(state_ini_mean,np.eye(3),N).T
#  generate reference trajectory
sIntegrator.evaluate(s=np.random.multivariate_normal(state_ini_mean,np.eye(3)), t0=0.0, tf=T)
state_ref = sIntegrator.get_states().T
time = sIntegrator.get_times()

#  take observations
obs_time = np.arange(obs_incre,T+obs_incre,obs_incre)
Y = np.zeros([3,len(obs_time)])
for count, i in enumerate(obs_time):
    time_index = int(i/dt)
    Y[:,count] = state_ref[:,time_index] + np.random.multivariate_normal(np.zeros(3),np.identity(3, float)).T

#  FPF
controller = FPF_controller(particle_num=N, sensor_gradient=grad_h, observation_dim=3, state_dim=3)
t = 0.0
n = 0
h = 0
state_all_mean[:,0] = np.mean(state_ini_part,axis=1)
state_all_x[0,:] = state_ini_part[0,:]
state_all_y[0,:] = state_ini_part[1,:]
state_all_z[0,:] = state_ini_part[2,:]
for count, i in enumerate(obs_time):
    if n == 0:
        x_n = state_ini_part
    
    state_pri = np.zeros((3,N))
    x_t = np.zeros((int(obs_incre/dt)-1,N))
    y_t = np.zeros_like(x_t)
    z_t = np.zeros_like(x_t)

    for j in range(N):
        Integrator.evaluate(s=x_n[:,j].copy(), t0=t, tf=i)
        state_pri[:,j] = Integrator.get_states().T[:,-1]
        x_t[:,j] = Integrator.get_states().T[0,1:-1]
        y_t[:,j] = Integrator.get_states().T[1,1:-1]
        z_t[:,j] = Integrator.get_states().T[2,1:-1]
        
    state_all_mean[0,h+1:h+int(obs_incre/dt)]=np.mean(x_t,axis=1)
    state_all_mean[1,h+1:h+int(obs_incre/dt)]=np.mean(y_t,axis=1)
    state_all_mean[2,h+1:h+int(obs_incre/dt)]=np.mean(z_t,axis=1)

    state_all_x[h+1:h+int(obs_incre/dt),:]=x_t
    state_all_y[h+1:h+int(obs_incre/dt),:]=y_t
    state_all_z[h+1:h+int(obs_incre/dt),:]=z_t

    l = 0

    while round(l,5) <= 1.0:
        K = controller.control_K(prior_state=state_pri)
        sigma = controller.control_sigma(prior_state=state_pri,control_K=K)
        for k in range(N):
            I = Y[:,count] - 1/2 * (state_pri[:,k]+np.mean(state_pri,axis=1))
            state_pri[:,k] += (K @ I + 1/2 * sigma) * d_lambda

        l += d_lambda

    state_post = state_pri.copy()
    x_n = state_post.copy()
    state_all_mean[:,h+int(obs_incre/dt)]=np.mean(state_post,axis=1)

    state_all_x[h+int(obs_incre/dt),:]=state_post[0,:]
    state_all_y[h+int(obs_incre/dt),:]=state_post[1,:]
    state_all_z[h+int(obs_incre/dt),:]=state_post[2,:]

    n += 1
    h += int(obs_incre/dt) 
    t += obs_incre

# RMSE
RMSE_t_x = np.sqrt(np.mean((state_all_x-state_ref[0,:].reshape((int(T/dt)+1,1)))**2,axis=1))
RMSE_t_y = np.sqrt(np.mean((state_all_y-state_ref[1,:].reshape((int(T/dt)+1,1)))**2,axis=1))
RMSE_t_z = np.sqrt(np.mean((state_all_z-state_ref[2,:].reshape((int(T/dt)+1,1)))**2,axis=1))

RMSE_mean = 1/3 * (RMSE_t_x + RMSE_t_y + RMSE_t_z)

print(f"Average RMSE from continuous discrete FPF={RMSE_mean}")

figure(1)
plot(np.linspace(0.,T,int(T/dt+1)),np.mean(state_all_x, axis=1), label='Posterior')
plot(np.linspace(0.,T,int(T/dt+1)),state_ref[0,:], label='Reference')
scatter(obs_time, Y[0,:], color='black', label='Observations')
xlabel('Time (s)')
ylabel('x_t')
legend()
show()