import os
import numpy as np
import pickle
from car_ros2.bag_utils import BagReader, synchronize_time
import matplotlib.pyplot as plt
import datetime
import math
from matplotlib.gridspec import GridSpec
from car_dataset import CarDataset
from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_planner import CAR_PLANNER_ASSETS_DIR

data_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
data_folder_path = os.path.join(CAR_FOUNDATION_DATA_DIR, f'{data_folder_prefix}-rosbag-dataset')

bag_path = "/home/gzh/Desktop/anycar/anycar/record_result_mppi_transformer/record_result_mppi_transformer_0.db3"

bag = BagReader(bag_path, full_state=True)

MAX_EPISODE_LENGTH = 2000

action_list = bag.bag_dict['/ackermann_command']
state_list = bag.bag_dict['/odometry_copy']
lateral_error_list = bag.bag_dict['/lateral_error']
ref_path = bag.bag_dict['/waypoint_list']


print("before sync", len(state_list))
action_list = action_list[10:]

actions = [a for t, a in action_list]
states = [s for t, s in state_list]
lateral_error = [s for t, s in lateral_error_list]

t_actions = np.array([t for t, a in action_list])
t_states = np.array([t for t, s in state_list])

path_x_list = []
path_y_list = []

track = np.loadtxt(os.path.join(CAR_PLANNER_ASSETS_DIR, "cuc_inside.csv"), delimiter=',', skiprows=1)

path_points = track[:, [0, 1]]
paht_x_list = track[:, 0]
paht_y_list = track[:, 1]

t_states_new, states_new = synchronize_time(t_states, t_actions, states)
t_lateral_error_new, lateral_error_new = synchronize_time(t_states, t_actions, lateral_error)


assert states_new.shape[1] == 9
dataset = CarDataset()

dataset.car_params['sim'] = 'car-real'

print("States shape", states_new.shape)
i = 0
for state, action in zip(states_new, actions):
    dataset.data_logs["xpos_x"].append(state[0])
    dataset.data_logs["xpos_y"].append(state[1])
    dataset.data_logs["xpos_z"].append(0.)
    #log orientation
    dataset.data_logs["xori_w"].append(state[5])
    dataset.data_logs["xori_x"].append(state[2])
    dataset.data_logs["xori_y"].append(state[3])
    dataset.data_logs["xori_z"].append(state[4])
    #log linear velocity
    dataset.data_logs["xvel_x"].append(state[6])
    dataset.data_logs["xvel_y"].append(state[7])
    dataset.data_logs["xvel_z"].append(0.)
    #log linear acceleration
    dataset.data_logs["xacc_x"].append(0.)
    dataset.data_logs["xacc_y"].append(0.)
    dataset.data_logs["xacc_z"].append(0.)
    #log angular velocity
    dataset.data_logs["avel_x"].append(0.)
    dataset.data_logs["avel_y"].append(0.)
    dataset.data_logs["avel_z"].append(state[8])

    dataset.data_logs["traj_x"].append(0.)
    dataset.data_logs["traj_y"].append(0.)

    dataset.data_logs["lap_end"].append(0)

    dataset.data_logs["throttle"].append(action[0])
    dataset.data_logs["steer"].append(action[1])
 
    i += 1
    
    if i == MAX_EPISODE_LENGTH:
        data_dir = os.path.join(data_folder_path)
            
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        dataset.data_logs["lap_end"][-1] = 1    
        now = datetime.datetime.now().isoformat(timespec='milliseconds')
        file_name = "log_" + str(now) + ".pkl"
        filepath = os.path.join(data_dir, file_name)
        for key, value in dataset.data_logs.items():
            dataset.data_logs[key] = np.array(value)

        with open(filepath, 'wb') as outp: 
            pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
        print("Saved Data to:", filepath)
        
        i = 0
        dataset.reset_logs()
        

for i in range(states_new.shape[0]-1):
    if states_new[i, 0] == states_new[i+1, 0]:
        print(i)

yaw_list = [
    math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)) * 57.3
    for w, x, y, z in zip(states_new[:, 5], states_new[:, 2], states_new[:, 3], states_new[:, 4])
]


if len(state_list) > 1000 :
    plot_num = 1000
else:
    plot_num = len(state_list)
    
# calculate lateral error rmse
lateral_error_rmse = np.sqrt(np.mean(np.array(lateral_error_new[:plot_num]) ** 2))

print("lateral error rmse = " + str(lateral_error_rmse))

fig = plt.figure(figsize=(8, 12))
gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1])

ax1 = fig.add_subplot(gs[0])
ax1.plot(paht_x_list, paht_y_list, 'g', label='path', marker='o')
ax1.plot(states_new[:plot_num, 0],states_new[:plot_num, 1], 'r', label='real', marker='o')
ax1.set_title("Trajectory")
ax1.legend()
ax1.grid()

ax2 = fig.add_subplot(gs[1])
ax2.plot(lateral_error_new[:plot_num], 'r', label='lateral error', marker='o')
ax2.set_title("Lateral Error")
ax2.legend()
ax2.grid()

ax3 = fig.add_subplot(gs[2])
ax3.plot(yaw_list[:plot_num], 'r', label='yaw', marker='o')
ax3.set_title("Yaw List")
ax3.legend()
ax3.grid()

ax4 = fig.add_subplot(gs[3])
ax4.plot(np.array(states_new[:plot_num, 8]), 'r', label='yawrate', marker='o')
ax4.set_title("Yaw Rate")
ax4.legend()
ax4.grid()

plt.tight_layout()
plt.show()
