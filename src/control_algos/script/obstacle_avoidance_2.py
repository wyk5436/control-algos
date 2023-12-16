###
# NOTE: This script does the same thing as "obstacle_avoidance.py", but does not use the inegrator class.
# This runs faster than the other one.
###
import numpy as np
from matplotlib.pyplot import *
from matplotlib.patches import Rectangle
from tqdm import tqdm

from control_algos.controller.MPPI import MPPI_controller
from control_algos.models.UAV import UAV_model

# Define parameters
traj_num = 5 # number of trajectories in total
MC_run_total = 10000
traj_fail = 0 # number of failed trajectories
x0 = np.array([-0.4, -0.4, 0., 0.]) # 4 states are: x, y, v_x, v_y
dt = 0.01
T = 7
eta = 0.67
d = 1
a = 1
g2 = 7
sigma2 = 0.01
l = sigma2/(1/a-1/g2)
k = 0.288
count_failure = 0
eps_t_all = np.zeros((2,MC_run_total))
G = np.zeros((4,2))
G[2,0] = 1
G[3,1] = 1
np.random.seed(1234)

xP = -0.5
yP = -0.5
A = 0.6
B = 0.6
xQ = xP+A
yQ = yP+B
xR1 = -0.3
yR1 = -0.4
xS1 = -0.2
yS1 = -0.25
xR2 = -0.3
yR2 = -0.15
xS2 = -0.1
yS2 = 0

# Create figure and axis
figure(2)
gca().set_xlim([xP-0.05, xQ+0.05])
gca().set_ylim([yP-0.05, yQ+0.05])
gca().set_aspect('equal', adjustable='box')
gca().set_xticks([i/10 for i in range(int(xP*10), int(xQ*10)+1)])
gca().set_yticks([i/10 for i in range(int(yP*10), int(yQ*10)+1)])
xlabel('$p_x$', fontsize=30, fontname='Arial')
ylabel('$p_y$', fontsize=30, fontname='Arial')
gca().tick_params(axis='both', which='major', labelsize=18)
gca().set_facecolor('white')
gca().spines['top'].set_linewidth(1)
gca().spines['right'].set_linewidth(1)
gca().spines['bottom'].set_linewidth(1)
gca().spines['left'].set_linewidth(1)

# Outer rectangle
outer_rect = Rectangle((xP, yP), xQ-xP, yQ-yP, facecolor='none', edgecolor='red', linewidth=1.5)
gca().add_patch(outer_rect)

# Inner rectangle 1
inner_rect1 = Rectangle((xR1, yR1), xS1-xR1, yS1-yR1, facecolor='red', edgecolor='black', linewidth=2.5, hatch='\\')
gca().add_patch(inner_rect1)

# Inner rectangle 2
inner_rect2 = Rectangle((xR2, yR2), xS2-xR2, yS2-yR2, facecolor='red', edgecolor='black', linewidth=2.5, hatch='\\')
gca().add_patch(inner_rect2)

# Draw starting and targeting positions
scatter(x0[0], x0[1], marker='*', s=500, edgecolor='yellow', facecolor='yellow', linewidth=2)
scatter(0, 0, marker='*', s=500, edgecolor='green', facecolor='green', linewidth=2)

# MPPI
EOM = UAV_model(k=k)
for traj in range(traj_num):
    print(f"Current trajectory={traj+1}")
    x = np.zeros((4, int(T/dt)+1))
    x[:,0] = x0.copy()
    safety_flag_traj = True
    cost = np.zeros(MC_run_total)
    noise = np.zeros((2, MC_run_total))
    xt = x0.copy()
    f = EOM.evaluate(stepsize=dt, time=0, states=xt)
    for count, t in tqdm(enumerate(np.linspace(0.0,T-dt,int((T-dt)/dt)+1)), total=len(np.linspace(0.0,T-dt,int((T-dt)/dt)+1)), desc="Processing", unit="time step"):
        eps_t_all_1 = np.random.normal(loc=0., scale=1.0, size=MC_run_total)
        eps_t_all_2 = np.random.normal(loc=0., scale=1.0, size=MC_run_total)

        controller = MPPI_controller(MC_run_total=MC_run_total,
                                     current_time=t, 
                                     final_time=T, 
                                     time_step=dt, 
                                     current_states=xt.copy(),
                                     k=k, 
                                     eta=eta, 
                                     d=d,
                                     l=l,
                                     a=a,
                                     g2=g2,
                                     xR1=xR1,
                                     xR2=xR2,
                                     xS1=xS1,
                                     xS2=xS2,
                                     yR1=yR1,
                                     yR2=yR2,
                                     yS1=yS1,
                                     yS2=yS2,
                                     xP=xP,
                                     xQ=xQ,
                                     yP=yP,
                                     yQ=yQ)
        for MC_run in range(MC_run_total):
            cost_i = controller.cost_function(eps_1=eps_t_all_1[MC_run], eps_2=eps_t_all_2[MC_run],state=xt.copy(),f=f.copy())
            cost[MC_run] = cost_i
        
        eps_t_all[0,:] = eps_t_all_1
        eps_t_all[1,:] = eps_t_all_2

        denom_i = np.exp(-cost/l)
        numer = eps_t_all @ denom_i
        denom = np.sum(denom_i)

        ut = np.identity(2) * (1-a/g2)**(-1) @ numer/(np.sqrt(dt)*denom)
        vt = -np.identity(2) * (g2/a-1)**(-1) @ numer/(np.sqrt(dt)*denom)

        xt[0] += f[0] * dt + G[0,:] @ ut * dt +G[0,:] @ vt * dt
        xt[1] += f[1] * dt + G[1,:] @ ut * dt +G[1,:] @ vt * dt
        xt[2] += f[2] * dt + G[2,:] @ ut * dt +G[2,:] @ vt * dt + 0.1 * np.random.normal(loc=0., scale=1.0) * np.sqrt(dt)
        xt[3] += f[3] * dt + G[3,:] @ ut * dt +G[3,:] @ vt * dt + 0.1 * np.random.normal(loc=0., scale=1.0) * np.sqrt(dt)

        x[:,count+1] = xt

        safety_flag_traj = controller.check_safety(current_states=xt)

        if safety_flag_traj == False:
            traj_fail += 1
            count_failure = count
            break

        f = EOM.evaluate(stepsize=dt, time=0, states=xt)

    if safety_flag_traj == True:
        figure(2)
        plot(x[0,:],x[1,:],'b-',linewidth=1)
    else:
        figure(2)
        plot(x[0,:count_failure],x[1,:count_failure],'k-',linewidth=1)
show()
