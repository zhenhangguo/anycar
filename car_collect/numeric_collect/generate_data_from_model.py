import datetime
import os
import ray
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from rich.progress import track
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R

from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_vehicle_parameters
from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN
from nuplan.planning.simulation.controller.motion_model.bobtail_trailer_model import BobtailTrailerModel

from generate_data_utils import *


# params
Use_Steer_Angle = True
Save_Data = True
Add_DeadZone_ThS = False
Back_Lash = True
Random_Steer_Offset = False
Fixed_steer_offset = 5 # deg
Small_steer_cmd = True
Small_max_steer_cmd = 30  # deg

Total_Data_num  = 50000
Simend = 2000


data_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
    
def car_orientation(yaw):
    ori1 = np.array([
        np.cos(yaw/2),
        0., 
        0., 
        np.sin(yaw/2), 
    ])
    r = R.from_euler('xyz', [0., 0., yaw])
    ori2 = r.as_quat()
    ori2 = np.array([ori2[3], ori2[0], ori2[1], ori2[2]])
    assert np.allclose(ori1, ori2)
    return ori1

def log_data(dataset: CarDataset, env: car_state, action):
        dataset.data_logs["xpos_x"].append(env.car_pos[0])
        dataset.data_logs["xpos_y"].append(env.car_pos[1])
        dataset.data_logs["xpos_z"].append(0.0)
        #log orientation
        dataset.data_logs["xori_w"].append(env.car_orientation[0])
        dataset.data_logs["xori_x"].append(env.car_orientation[1])
        dataset.data_logs["xori_y"].append(env.car_orientation[2])
        dataset.data_logs["xori_z"].append(env.car_orientation[3])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(env.car_lin_vel[0])
        dataset.data_logs["xvel_y"].append(env.car_lin_vel[1])
        dataset.data_logs["xvel_z"].append(0.0)
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.car_lin_acc[0])
        dataset.data_logs["xacc_y"].append(env.car_lin_acc[1])
        dataset.data_logs["xacc_z"].append(0.0)
        #log angular velocity
        dataset.data_logs["avel_x"].append(0.0)
        dataset.data_logs["avel_y"].append(0.0)
        dataset.data_logs["avel_z"].append(env.car_ang_vel[2])

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1])
        
        dataset.data_logs["vehicle_yaw"].append(env.car_yaw)

def plot_data_result(dataset: CarDataset, t):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].plot(dataset.data_logs["xpos_x"][:300], dataset.data_logs["xpos_y"][:300], label='position', marker='o', markersize=3)
    axs[0, 0].legend()
    axs[0, 0].axis('equal')

    axs[0, 1].plot(dataset.data_logs["xvel_x"][:300], label='data vx')
    axs[0, 1].plot(dataset.data_logs["xvel_y"][:300], label='data vy')
    axs[0, 1].legend()
    
    axs[1, 0].plot(np.array(dataset.data_logs["vehicle_yaw"][:300]) * 57.3, label='data yaw')
    axs[1, 0].legend()
    
    axs[1, 1].plot(np.array(dataset.data_logs["avel_z"][:300]) * 57.3, label='data yawrate')
    axs[1, 1].plot(np.array(dataset.data_logs["steer"][:300]) * 57.3 * 0.1, label='steer angle * 0.1')
    axs[1, 1].legend()

    fig.tight_layout()
    save_path = os.path.join("/disk1/collect_data_from_anycar/figure/", "result" + str(t+1))
    fig.savefig(save_path, format="png")
    # plt.show()
    plt.close()

def update_vehicle_state(vehicle_state, state):
    # pose position
    vehicle_state.car_pos[0] = state.rear_axle.x
    vehicle_state.car_pos[1] = state.rear_axle.y

    # pose orientation
    vehicle_state.car_orientation  = car_orientation(state.rear_axle.heading)

    vehicle_state.car_ang_vel[2] = state.dynamic_car_state.angular_velocity
    vehicle_state.car_yaw = state.rear_axle.heading

    # pose linear vel
    vehicle_state.car_lin_vel[0]  = state.dynamic_car_state.rear_axle_velocity_2d.x
    vehicle_state.car_lin_vel[1]  = state.dynamic_car_state.rear_axle_velocity_2d.y

    vehicle_state.car_lin_acc[0]  = state.dynamic_car_state.rear_axle_acceleration_2d.x
    vehicle_state.car_lin_acc[1]  = state.dynamic_car_state.rear_axle_acceleration_2d.y
    
    large_yawrate = abs(vehicle_state.car_ang_vel[2]) > 1e3
    unfit_speed = vehicle_state.car_lin_vel[0] > 30.0 or vehicle_state.car_lin_vel[0] < 5.0
    if large_yawrate or unfit_speed:
        return False
    else:
        return True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clamp_and_smooth_cmd(max_cmd, cmd):
    mean = 0.5
    
    return (sigmoid(5 * cmd) - mean) * max_cmd

@ray.remote   
def rollout(params):
    id, time_steps, datadir = params
    tic = time.time()
    # get vehicle model 
    VEHICLE = get_vehicle_parameters()
    dataset = CarDataset()
    
    # vehicle_ratio
    vehicle_ratio = 25

    heading = np.random.uniform(-3.14, 3.14)

    lon_speed = np.random.uniform(5, 25)

    if Add_DeadZone_ThS or Back_Lash:
        tire_steering_rate = 0.0
    else:
        tire_steering_rate = np.random.uniform(-0.5, 0.5)

    dynamic_car_state = DynamicCarState.build_from_rear_axle(
        VEHICLE.rear_axle_to_center,
        rear_axle_velocity_2d=StateVector2D(lon_speed, 0),
        rear_axle_acceleration_2d=StateVector2D(0, 0),
        tire_steering_rate=tire_steering_rate
    )

    car_footprint = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=heading), vehicle_parameters=VEHICLE
    )

    init_front_wheel_angle = 0.2

    if Add_DeadZone_ThS or Back_Lash:
        steering_angle = np.random.uniform(-init_front_wheel_angle, init_front_wheel_angle) * vehicle_ratio
        steering_angle_offset = np.random.uniform(-0.01, 0.01) * vehicle_ratio
        
        if Small_steer_cmd:
            steering_angle = clamp_and_smooth_cmd(Small_max_steer_cmd / 57.3, steering_angle)

        if Add_DeadZone_ThS:
            front_wheel_angle = steer_hysteresis_actuator(lon_speed, steering_angle, steering_angle_offset)
        if Back_Lash:
            steer_backlash = backlash(8.0 / 57.3)   #backlash(np.random.uniform(8.0))
            front_wheel_angle = steer_backlash.calculate(steering_angle) / vehicle_ratio
    else:
        if Small_steer_cmd:
            steering_angle = clamp_and_smooth_cmd(Small_max_steer_cmd / 57.3 / vehicle_ratio, steering_angle)
        front_wheel_angle = np.random.uniform(-init_front_wheel_angle, init_front_wheel_angle)
        steering_angle_offset = np.random.uniform(-0.01, 0.01)  # actual front wheel angle offset

    if not Random_Steer_Offset:
        steering_angle_offset = 0.0

    state = EgoState(
        car_footprint,
        dynamic_car_state,
        tire_steering_angle=front_wheel_angle,
        is_in_auto_mode=True,
        time_point=TimePoint(0),
    )

    motion_model = BobtailTrailerModel(VEHICLE)

    track_change_time = time_steps
    tstep = 0.05

    # add PID for stable lateral state
    if Small_steer_cmd:
        kp = np.random.uniform(0.05, 0.5)
    else:
        kp = np.random.uniform(0.5, 1.5)

    # random walker
    randcontrol = RandWalkController(track_change_time, 0.0, 0.0, lon_speed, heading, front_wheel_angle, tstep)

    for t in track(range(time_steps)):
        action = randcontrol.get_control(t%track_change_time)

        action[1] = action[1] + state.dynamic_car_state.angular_velocity * -kp

        # calculate steer angle
        if Add_DeadZone_ThS  or Back_Lash:
            steering_angle = steering_angle + action[1] * vehicle_ratio * tstep
            
            if Small_steer_cmd:
                steering_angle = clamp_and_smooth_cmd(Small_max_steer_cmd / 57.3, steering_angle)
            
            if Add_DeadZone_ThS:
                front_wheel_angle_rate = (steer_hysteresis_actuator(state.dynamic_car_state.rear_axle_velocity_2d.x, steering_angle, steering_angle_offset) - front_wheel_angle) / tstep
                front_wheel_angle = steer_hysteresis_actuator(state.dynamic_car_state.rear_axle_velocity_2d.x, steering_angle, steering_angle_offset)
            if Back_Lash:
                new_front_wheel_angle = steer_backlash.calculate(steering_angle) / vehicle_ratio
                front_wheel_angle_rate = (new_front_wheel_angle - front_wheel_angle) / tstep
                front_wheel_angle = front_wheel_angle + front_wheel_angle_rate * tstep
        else:
            steering_angle = state._tire_steering_angle + steering_angle_offset  # actual front wheel angle
            
            if Small_steer_cmd:
                steering_angle = clamp_and_smooth_cmd(Small_max_steer_cmd / 57.3 / vehicle_ratio, steering_angle)
            
            front_wheel_angle_rate = action[1]

        # update sim vehicle state
        ideal_dynamic_state = state.dynamic_car_state
        ideal_dynamic_state.tire_steering_rate = front_wheel_angle_rate
        
        state = motion_model.propagate_state(state, ideal_dynamic_state, TimePoint(tstep * 1e6))
        state.dynamic_car_state.rear_axle_acceleration_2d.x = action[0]

        # update data into dataset
        vehicle_state = car_state()
        success = update_vehicle_state(vehicle_state, state)
        
        if not success:
            return

        if Use_Steer_Angle:
            log_data(dataset, vehicle_state, [action[0], steering_angle + Fixed_steer_offset / 57.3])
        else:
            log_data(dataset, vehicle_state, action)

        # check end timestep
        if ((t+1) % track_change_time == 0) and Save_Data:
            dataset.data_logs["lap_end"][-1] = 1 # log the end of the lap
            randcontrol.reset_controller()
        
            # save data into files
            now = datetime.datetime.now().isoformat(timespec='milliseconds')
            file_name = "log_" + str(id) + '_' + str(now) + ".pkl"
            filepath = os.path.join(datadir, file_name)
            
            for key, value in dataset.data_logs.items():
                dataset.data_logs[key] = np.array(value)
            
            with open(filepath, 'wb') as outp: 
                pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
            
            print("Saved Data to:", filepath)
            # plot_data_result(dataset, id)
            dataset.reset_logs()

    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)

if __name__ == "__main__":
    
    # simulation params
    simend = Simend
    episodes = Total_Data_num

    data_dir = os.path.join('/disk1/collect_data_from_anycar/', f'{data_folder_prefix}-nuplan-dynamic-model')
    os.makedirs(data_dir, exist_ok=True)

    futures = [rollout.remote((i, simend, data_dir)) for i in range(episodes)]
    done = [] 

    # Function to track progress
    def track_progress(futures):
        while len(futures) > 0:
            done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
            for _ in done:
                yield

    # Use rich.progress.track to display progress
    for _ in track(track_progress(futures), 
                description="Collecting data...", total=len(futures),disable=False):
        pass

    # Collect the results from workers
    results = ray.get(futures + done)