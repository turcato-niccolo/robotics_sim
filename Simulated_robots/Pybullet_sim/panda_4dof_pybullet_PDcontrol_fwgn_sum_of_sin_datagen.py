"""
Pybullet simulation of the Franka Emika Panda robot restricted to 4DOF,
for the generation of training and test dataset.

Author: Giulio Giacomuzzo (giulio.giacomuzzo@phd.unipd.it)
Edited: Niccolo' Turcato (niccolo.turcato@studenti.unipd.it)
"""
# %% Preamble

from numpy.core.function_base import linspace
from pandas.core.frame import DataFrame
import pybullet
import pybullet_data
import math
import datetime
import numpy as np
from numpy import random
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import pickle as pkl
import time
import argparse
from datagen import *

# %% Set command line args

parser = argparse.ArgumentParser('Franka Emika Panda (4DOF) PD vel controller')

parser.add_argument('-saving_path',
                    type=str,
                    default='./FE_panda/data/',
                    help='destination folder for the generated files')

parser.add_argument('-robot_name',
                    type=str,
                    default='FE_panda4dof_pybul',
                    help='name of the simulated robot')

parser.add_argument('-robot_urdf',
                    type=str,
                    default='./FE_panda/models/panda_arm_4dof.urdf',
                    help='path to the robot urdf file')

locals().update(vars(parser.parse_known_args()[0]))

# Flag save: if true save the generated data
flg_save = True

# %% GET THE PYBULLET ENVIRONMENT

# Start the pybullet client and server processes
pybullet.connect(pybullet.GUI)

# Add the path to the example models
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set the gravity vector -- by default there is no gravity
pybullet.setGravity(0, 0, -9.81)

# Spawn a ground plane
planeId = pybullet.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True)

# Spawn the robot
pandaID = pybullet.loadURDF(robot_urdf, [0, 0, 0], flags=pybullet.URDF_USE_SELF_COLLISION, useFixedBase=True)

# Print useful debug info
print('Plane Body ID = ', planeId)
print('Robot Body ID = ', pandaID)

# reset the position of the robot base
pybullet.resetBasePositionAndOrientation(pandaID, [0, 0, 0], [0, 0, 0, 1])

# Store the number of joints
num_joints = pybullet.getNumJoints(pandaID)  # -2

# Set the dynamical properties for each link (friction and damping coeffs)
for i in range(num_joints):
    pybullet.changeDynamics(bodyUniqueId=pandaID,
                            linkIndex=i,
                            lateralFriction=0.0,
                            spinningFriction=0.0,
                            rollingFriction=0.0,
                            linearDamping=0.0,
                            angularDamping=0.0,
                            jointDamping=0.0)

# Disable Default constrained based motors
for i in range(num_joints):
    pybullet.setJointMotorControl2(pandaID, i, pybullet.POSITION_CONTROL, targetPosition=0, force=0)


# Define joint operational ranges
ll = [-2.9, -1.8, -2.9, -3.0] #[rad] -- last value is for joint 8 which is fixed
ul = [2.9, 1.8, 2.9, 0.0] #[rad] -- last value is for joint 8 which is fixed
ranges = [ul[i] - ll[i] for i in range(num_joints)]
mean_ranges = np.array([(ul[i] + ll[i])/2 for i in range(num_joints)])

# Set the robot rest pose
rest_pose = np.zeros((num_joints, 1))

# Set the joint max torque
max_torques = [87]*4 # Nm

max_dq = [2.1750]*4 + [2.6100]*3
max_ddq = [15, 7.5, 10, 12.5, 15, 20, 20]
max_dddq = [7500, 3750, 5000, 6250, 7500, 10000, 10000]

scaling = 1
positionGains = [1 * scaling] * (num_joints)
velocityGains = [10 * scaling] * (num_joints)

# %%
# SIMULATION

# Set the simulation sampling time
dt = 0.001  # 1ms, default to 1./240 s
pybullet.setTimeStep(dt)

# Initialize a random number generator for repeatability
rng = random.default_rng(seed=10)

# Remove the first part of the simulation due to numerical inaccuracies
record_delay = 1.2  # seconds

# Perform the simulation twice: one for train and one for test
for run in ['train', 'test']:

    # Set the number of simulation steps
    num_steps = 50000 + int(record_delay / dt)  # 50 s

    if run == 'train':
        num_steps += 2
        # Get the reference trajectories: filtered wgn
        fc = 0.5  # [Hz] cutting frequency of the filter
        std_scalings = 1.4 * np.diag(np.array([1, 1, 1, 1]))
        std_vec = np.dot(std_scalings, np.array(ranges))
        # std_vec = 3*np.array([rang for rang in ranges[:-1]])
        q_ref, dq_ref, ddq_ref = get_filtered_wgn_trj(num_steps, dt, num_joints, rng, mean_ranges, std_vec, fc)
    else:
        max_freq = 0.5
        num_freq = 10
        max_q = np.abs(np.array(ranges) / 2)
        q_ref, dq_ref, ddq_ref = get_sum_of_sinusoids_trj(num_steps, dt, num_joints, rng, max_freq, num_freq, max_q,
                                                          mean_ranges, max_dq, max_ddq)
    # Update number of simulation steps: numerical derivatives require to discard intial and final samples
    num_steps = len(q_ref)

    # Set the initial position on the reference initial position
    for i in range(num_joints):
        pybullet.resetJointState(pandaID, i, q_ref[0, i])
        # pybullet.resetJointState(pandaId, i, 0.0)

    q = []
    dq = []
    tau = []
    M = []
    g = []
    c = []
    m = []

    M_k_prev = None
    for k in range(num_steps):
        # Initialize list to store the current pos vel torque and Mass matrix
        q_k = []
        dq_k = []
        tau_k = []
        M_k = []
        # Get current states from pybullet
        js = pybullet.getJointStates(pandaID, list(range(num_joints)))

        q_k = [x[0] for x in js]
        dq_k = [x[1] for x in js]
        tau_k = [x[-1] for x in js]

        # q_k = [js[i][0] for i in range(num_joints)]
        # dq_k = [js[i][1] for i in range(num_joints)]

        # tau_k = [js[i][-1] for i in range(num_joints)]

        g_k = pybullet.calculateInverseDynamics(pandaID, q_k, [0] * (num_joints), [0] * (num_joints))

        c_k = pybullet.calculateInverseDynamics(pandaID, q_k, dq_k, [0] * (num_joints))
        c_k = [c_k[i] - g_k[i] for i in range(num_joints)]

        q.append(q_k)
        dq.append(dq_k)
        tau.append(tau_k)
        # Get the current inertial matrix from pybullet
        M_k = pybullet.calculateMassMatrix(pandaID, q_k)
        M_k_prev = M_k
        M.append(M_k)
        g.append(g_k)
        c.append(c_k)

        # Send the control mode and the targets to each joint
        pybullet.setJointMotorControlArray(pandaID,
                                           list(range(num_joints)),
                                           controlMode=pybullet.POSITION_CONTROL,
                                           targetPositions=list(q_ref[k]),
                                           targetVelocities=list(dq_ref[k]),
                                           forces=max_torques)
        # positionGains=positionGains,
        # velocityGains=velocityGains)
        # Step the simulation
        pybullet.stepSimulation()

    # Convert q dq tau and M into numpy arrays
    if run == 'train':
        q_ref_tr = q_ref[int(record_delay / dt):, :]
        dq_ref_tr = dq_ref[int(record_delay / dt):, :]
        q_tr = np.array([np.array(q_k) for q_k in q])  # [int(record_delay/dt):,:]
        dq_tr = np.array([np.array(dq_k) for dq_k in dq])  # [int(record_delay/dt):,:]

        ddq_tr = np.zeros(dq_tr.shape)
        ddq_tr[0, :] = (dq_tr[1, :] - dq_tr[0, :]) / dt
        ddq_tr[1:-1, :] = (dq_tr[2:, :] - dq_tr[:-2]) / (2 * dt)
        ddq_tr[-1, :] = (dq_tr[-1, :] - dq_tr[-2, :]) / dt

        tau_tr = np.array([np.array(tau_k) for tau_k in tau])[int(record_delay / dt):, :]
        g_tr = np.array([np.array(g_k) for g_k in g])[int(record_delay / dt):, :]
        c_tr = np.array([np.array(c_k) for c_k in c])[int(record_delay / dt):, :]
        m_tr = np.zeros(ddq_tr.shape)
        M_tr = np.zeros([num_steps, num_joints, num_joints])
        for i in range(num_steps):
            M_tr[i, :, :] = np.array(M[i])
            m_tr[i, :] = np.matmul(M_tr[i], ddq_tr[i])

        M_tr = M_tr[int(record_delay / dt):, :, :]
        m_tr = m_tr[int(record_delay / dt):, :]
        tau_tr_reconstr = g_tr + c_tr + m_tr
        q_tr = q_tr[int(record_delay / dt):, :]
        dq_tr = dq_tr[int(record_delay / dt):, :]
        ddq_tr = ddq_tr[int(record_delay / dt):, :]

        tau_tr_std = [np.std(tau_tr[:, i]) for i in range(num_joints)]
        tr_noise_std = np.array(tau_tr_std) * 5e-2

        tau_tr_noised = noising_signals(tau_tr, tr_noise_std, [0] * (num_joints))

    else:
        q_ref_test = q_ref[int(record_delay / dt):, :]
        dq_ref_test = dq_ref[int(record_delay / dt):, :]
        q_test = np.array([np.array(q_k) for q_k in q])  # [int(record_delay/dt):,:]
        dq_test = np.array([np.array(dq_k) for dq_k in dq])  # [int(record_delay/dt):,:]
        ddq_test = np.zeros(dq_test.shape)

        ddq_test[0, :] = (dq_test[1, :] - dq_test[0, :]) / dt
        ddq_test[1:-1, :] = (dq_test[2:, :] - dq_test[:-2]) / (2 * dt)
        ddq_test[-1, :] = (dq_test[-1, :] - dq_test[-2, :]) / dt

        tau_test = np.array([np.array(tau_k) for tau_k in tau])[int(record_delay / dt):, :]
        g_test = np.array([np.array(g_k) for g_k in g])[int(record_delay / dt):, :]
        c_test = np.array([np.array(c_k) for c_k in c])[int(record_delay / dt):, :]

        M_test = np.zeros([num_steps, num_joints, num_joints])
        m_test = np.zeros(ddq_test.shape)
        for i in range(num_steps):
            M_test[i, :, :] = np.array(M[i])
            m_test[i, :] = np.matmul(M_test[i], ddq_test[i])

        M_test = M_test[int(record_delay / dt):, :, :]
        m_test = m_test[int(record_delay / dt):, :]

        tau_test_reconstr = g_test + c_test + m_test
        q_test = q_test[int(record_delay / dt):, :]
        dq_test = dq_test[int(record_delay / dt):, :]
        ddq_test = ddq_test[int(record_delay / dt):, :]

# Disconnect Pybullet
pybullet.disconnect()

# Build the times vector
t = linspace(0, len(q_tr) * dt, len(q_tr))

# %%
# Save simulation data if flg_save is True
if flg_save:
    # Create the names list for the dataframe cols
    q_ref_names = ['q_ref_' + str(k + 1) for k in range(0, num_joints)]
    dq_ref_names = ['dq_ref_' + str(k + 1) for k in range(0, num_joints)]
    q_names = ['q_' + str(k + 1) for k in range(0, num_joints)]
    dq_names = ['dq_' + str(k + 1) for k in range(0, num_joints)]
    ddq_names = ['ddq_' + str(k + 1) for k in range(0, num_joints)]
    tau_names = ['tau_' + str(k + 1) for k in range(0, num_joints)]
    m_names = ['m_' + str(k + 1) for k in range(0, num_joints)]
    c_names = ['c_' + str(k + 1) for k in range(0, num_joints)]
    g_names = ['g_' + str(k + 1) for k in range(0, num_joints)]
    tau_noised_names = ['tau_noised_' + str(k + 1) for k in range(0, num_joints)]
    data_names = q_ref_names + dq_ref_names + q_names + dq_names + ddq_names + tau_names + m_names + c_names + g_names

    # Create the dataframes
    data_frame_tr = pd.DataFrame(
        data=np.concatenate([q_ref_tr, dq_ref_tr, q_tr, dq_tr, ddq_tr, tau_tr, m_tr, c_tr, g_tr, tau_tr_noised],
                            axis=1), columns=data_names + tau_noised_names)
    data_frame_test = pd.DataFrame(
        data=np.concatenate([q_ref_test, dq_ref_test, q_test, dq_test, ddq_test, tau_test, m_test, c_test, g_test],
                            axis=1), columns=data_names)

    # Store the dataframe as a pkl file
    pkl.dump(data_frame_tr, open(saving_path + robot_name + '_fwgn_tr.pkl', 'wb'))
    pkl.dump(data_frame_test, open(saving_path + robot_name + '_sum_of_sin_test.pkl', 'wb'))

# %%
# Plot ref trajectories
plt.figure()
plt.suptitle('Pos references')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, q_ref_tr[:, i], label=r'$q_{ref,tr}$')
    plt.plot(t, q_ref_test[:, i], label=r'$q_{ref,test}$')
    plt.plot(t, ul[i] * np.ones([len(t), ]), '-r')
    plt.plot(t, ll[i] * np.ones([len(t), ]), '-r')
    # plt.plot(t,dq_ref_tr[:,i],label=r'$\dot{q}_{ref,tr}$')
    plt.ylabel(r'$q_{ref}' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')
# plt.show()

# Plot actual positions
plt.figure()
plt.suptitle('Acutal pos')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, q_tr[:, i], label=r'$q_{tr}$')
    plt.plot(t, q_test[:, i], label=r'$q_{test}$')
    # plt.plot(t,ul[i]*np.ones([len(t),]), '-r')
    # plt.plot(t,ll[i]*np.ones([len(t),]), '-r')
    # plt.plot(t,dq[:,i],label='$\dot{q}$')
    plt.ylabel(r'$q_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')
# plt.show()

# Plot tracking errors
plt.figure()
plt.suptitle('Pos tracking Errors')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, q_ref_tr[:, i] - q_tr[:, i], label=r'$\tilde{q}_{tr}$')
    plt.plot(t, q_ref_test[:, i] - q_test[:, i], label=r'$\tilde{q}_{test}$')
    # plt.plot(t,ul[i]*np.ones([len(t),]), '-r')
    # plt.plot(t,ll[i]*np.ones([len(t),]), '-r')
    # plt.plot(t,dq[:,i],label='$\dot{q}$')
    plt.ylabel(r'$\tilde{q}_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')
# plt.show()

# Plot velocities
plt.figure()
plt.suptitle('Velocities')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, dq_tr[:, i], label=r'$\dot{q_{tr}}$')
    plt.plot(t, dq_test[:, i], label=r'$\dot{q_{test}}$')
    # plt.plot(t,dq_ref[:,i],label='$\dot{q}$')
    plt.ylabel('$\dot{q}_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# Plot accelerations
plt.figure()
plt.suptitle('Accelerations')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, ddq_tr[:, i], label=r'$\ddot{q_{tr}}$')
    plt.plot(t, ddq_test[:, i], label=r'$\ddot{q_{test}}$')
    # plt.plot(t,dq_ref[:,i],label='$\dot{q}$')
    plt.ylabel('$\ddot{q}_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# Plot applied torques
plt.figure()
plt.suptitle('Applied torques')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, tau_tr[:, i], label=r'$\tau_{tr}$')
    plt.plot(t, tau_test[:, i], label=r'$\tau_{test}$')
    # plt.plot(t,ul[i]*np.ones([len(t),]), '-r')
    # plt.plot(t,ll[i]*np.ones([len(t),]), '-r')
    # plt.plot(t,dq[:,i],label='$\dot{q}$')
    plt.ylabel(r'$\tau_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

plt.figure()
plt.suptitle('Noised torques')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, tau_tr_noised[:, i], label=r'$\tau\_noised_{tr}$')
    plt.plot(t, tau_tr[:, i], label=r'$\tau_{tr}$')
    plt.ylabel(r'$\tau_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# Plot Gravitational torques
plt.figure()
plt.suptitle('Applied Gravitational torques')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, g_tr[:, i], label=r'$g\_{tr}$')
    plt.plot(t, g_test[:, i], label=r'$g\_{test}$')
    plt.ylabel(r'$g\_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# Plot Coriolis torques
plt.figure()
plt.suptitle('Applied Coriolis torques')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, c_tr[:, i], label=r'$c\_{tr}$')
    plt.plot(t, c_test[:, i], label=r'$c\_{test}$')
    plt.ylabel(r'$c\_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# Plot Inertial torques
plt.figure()
plt.suptitle('Applied Inertial torques')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, m_tr[:, i], label=r'$m\_{tr}$')
    plt.plot(t, m_test[:, i], label=r'$m\_{test}$')
    plt.ylabel(r'$m\_' + str(i + 1) + '$')
    plt.grid()
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# Plot Error torques
plt.figure()
plt.suptitle('Error torques - torques reconstructed from decomposition')
for i in range(num_joints):
    plt.subplot(4, 2, i + 1)
    plt.plot(t, (tau_tr[:, i] - tau_tr_reconstr[:, i]), label=r'$e\_{tr}$')
    plt.plot(t, (tau_test[:, i] - tau_test_reconstr[:, i]), label=r'$e\_{test}$')
    plt.ylabel(r'$e\_' + str(i + 1) + '$')
    plt.grid()
    plt.suptitle('Error percentages in torques reconstructed from decomposition')
    print(np.mean(np.abs(tau_tr[:, i] - tau_tr_reconstr[:, i])) / np.max(np.abs(tau_tr[:, i])))
    print(np.mean(np.abs(tau_test[:, i] - tau_test_reconstr[:, i])) / np.max(np.abs(tau_test[:, i])))
plt.subplot(4, 2, 7)
plt.legend(bbox_to_anchor=(1.05, 0.90), loc='upper left')

# %%
# Control performance evaluation

# Cumulative tracking error
print('\n ------- Performance evaluation-------')
print('Cumulative tracking error:')
track_err_i = 0
track_err_tot = 0
for i in range(num_joints):
    track_err_i = (np.sum(np.abs(q_tr[:, i] - q_ref_tr[:, i])) + np.sum(np.abs(q_test[:, i] - q_ref_test[:, i]))) / 2
    track_err_tot += track_err_i
    print('    Joint ' + str(i) + ':', track_err_i)
print('Tot: ', track_err_tot)
print('-------------------------------------')

# Show plots
plt.show()
