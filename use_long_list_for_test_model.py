import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from scipy.spatial.transform import Rotation as R
import jax
from car_foundation.jax_models import JaxTransformerDecoder
import orbax.checkpoint
import pickle
import copy

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_vehicle_parameters
from nuplan.planning.simulation.controller.motion_model.bobtail_trailer_model import BobtailTrailerModel
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from generate_data_utils import *
from verify_utils import *

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

def update_vehicle_state(state):
    vehicle_state = car_state()
    
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
    
    return vehicle_state

VEHICLE = get_vehicle_parameters()
SAMPLING_TIME = TimePoint(1000000)

lon_speed = 20.0
dynamic_car_state = DynamicCarState.build_from_rear_axle(
    VEHICLE.rear_axle_to_center,
    rear_axle_velocity_2d=StateVector2D(lon_speed, 0.0),
    rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
    tire_steering_rate=0.0,
)
car_footprint = CarFootprint.build_from_rear_axle(
    rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0), vehicle_parameters=VEHICLE
)
state = EgoState(
    car_footprint,
    dynamic_car_state,
    tire_steering_angle=0.0,
    is_in_auto_mode=True,
    time_point=TimePoint(0),
)

motion_model = BobtailTrailerModel(VEHICLE)

car_state_list = []
action_1 = []

tstep = 0.05
sampling_time = TimePoint(tstep * 1e6)
input_output_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "UT_for_4thOrderModel_v20.npy")
input_output_array = np.load(input_output_file) * 0.1
input_array = input_output_array[0]
input_array = np.tile(input_array, 2)
lon_speed = 20.0
tire_steering_rate = np.diff(input_array).tolist()
tire_steering_rate += [0.0, 0.0]
for rate in tire_steering_rate:
    ideal_dynamic_state = DynamicCarState.build_from_rear_axle(
        VEHICLE.rear_axle_to_center,
        rear_axle_velocity_2d=StateVector2D(lon_speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_rate=rate / tstep,
    )
    action_1.append(rate / tstep)
    state = motion_model.propagate_state(state, ideal_dynamic_state, sampling_time)

    vehicle_state = update_vehicle_state(state)
    car_state_list.append(copy.deepcopy(vehicle_state))

Save_Data = True
data_dir = os.path.join('/disk1/collect_data_from_anycar/', f'{data_folder_prefix}-long-list-dynamic-model')
os.makedirs(data_dir, exist_ok=True)
if Save_Data:
    dataset_list = []
    need_num = 301
    pkl_num = int((len(car_state_list) - need_num) / 5)
    for idx in range(pkl_num):
        dataset = CarDataset()
        for i in range(need_num):
            real_idx = idx * 5 + i
            log_data(dataset, car_state_list[real_idx], [action_1[real_idx], 0.0])
        dataset.data_logs["lap_end"][-1] = 1
        dataset_list.append(dataset)
    
        # save data into files
        now = datetime.datetime.now().isoformat(timespec='milliseconds')
        file_name = "log_" + str(real_idx) + '_' + str(now) + ".pkl"
        filepath = os.path.join(data_dir, file_name)
    
        for key, value in dataset.data_logs.items():
            dataset.data_logs[key] = np.array(value)
            
        with open(filepath, 'wb') as outp: 
            pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
        
        dataset.reset_logs()
    
    print("Simulation Complete!")


pass
# load model
model_path = "/home/gzh/Desktop/anycar/anycar/car_foundation/car_foundation/models/2024-11-14-model_checkpoint/"
fig_result_path = '/home/gzh/Desktop/anycar/anycar/model_test_fig_long_list'
model_checkpint = 400

history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False
binary_mask = False
ATTACK = False  # verify data will not add noise

batch_size = 1024
state_dim = 6
action_dim = 2
latent_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1

rng = jax.random.PRNGKey(3407)
rng, params_rng = jax.random.split(rng)
rng, dropout_rng = jax.random.split(rng)
init_rngs = {'params': params_rng, 'dropout': dropout_rng}
global_rngs = init_rngs

jax_history_input = jnp.ones((batch_size, history_length-1, state_dim + action_dim), dtype=jnp.float32)
jax_history_mask = jnp.ones((batch_size, (history_length-1) * 2 - 1), dtype=jnp.float32)
jax_prediction_input = jnp.ones((batch_size, prediction_length, action_dim), dtype=jnp.float32)
jax_prediction_mask = jnp.ones((batch_size, prediction_length), dtype=jnp.float32)

model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length - 1, prediction_length, jnp.bfloat16, name='decoder')
val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(create=False, save_interval_steps=20)
checkpoint_manager = orbax.checkpoint.CheckpointManager(model_path, orbax_checkpointer)
raw_restored = checkpoint_manager.restore(os.path.join(model_path, f"{model_checkpint}"),"default")
val_collect['params'] = raw_restored['model']['params']

input_mean = jnp.array(raw_restored['input_mean'])
input_std = jnp.array(raw_restored['input_std'])

# def val_episode(var_collect, episode_num, rngs):
#     episode = test_dataset.get_episode(episode_num)
#     episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
#     batch = episode[:, :, :-1]
#     history, action, y, action_padding_mask = test_dataset[episode_num:episode_num+1]
#     history = jnp.array(history.numpy())
#     action = jnp.array(action.numpy())
#     y = jnp.array(y.numpy())
#     action_padding_mask = jnp.array(action_padding_mask.numpy())
#     predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs, input_mean, input_std, model)
#     return np.array(predicted_states)


Load_Data = False
if Load_Data:
    pkl_num = int((len(car_state_list) - 300) / 10)
    for idx in range(pkl_num):

        pass

# plt.plot(vehicle_state_x, vehicle_state_y)
# plt.show()
