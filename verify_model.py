import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import os
import glob
import torch
import matplotlib.pyplot as plt
import orbax.checkpoint

from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR
from car_foundation.dataset import MujocoDataset
from car_foundation.jax_models import JaxTransformerDecoder

from verify_utils import *

Save_Fig = True

model_path = "/disk1/collect_data_from_anycar/temp_verify_backlash_model/2025-01-14T18:44:21.817-model_checkpoint"
# dataset_path = '/disk1/collect_data_from_anycar/check_data/from_data_params_8'  #10 pkl
dataset_path = '/disk1/collect_data_from_anycar/New_demo/temp_check_data'  #10 pkl
# dataset_path = "/disk1/collect_data_from_anycar/data_from_bag/data_use_steer_angle/pde-a1"
# dataset_path = "/disk1/collect_data_from_anycar/data_from_bag/data_use_steer_angle/pdb-c11"
Long_Path_sim = False
fig_result_path = '/home/gzh//anycar/model_test_result_fig'

model_checkpint = 200

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
USE_ZERO_POINT=True

dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
test_dataset = MujocoDataset(dataset_files, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT)

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

print("input_mean = " + str(input_mean))
print("input_std = " + str(input_std))

print("model_hash = " + str(model_hash(val_collect['params'])))

def val_episode(var_collect, episode_num, rngs):
    episode = test_dataset.get_episode(episode_num)
    episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask = test_dataset[episode_num:episode_num+1]
    history = jnp.array(history.numpy())
    action = jnp.array(action.numpy())
    y = jnp.array(y.numpy())
    action_padding_mask = jnp.array(action_padding_mask.numpy())
    predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs, input_mean, input_std, model)
    return np.array(predicted_states)

# plot final result
def plot_final_result(data_num, rmse_dict):

    for epoch in range(data_num):
        
        if epoch + 1 >= len(test_dataset):
            continue
        
        predicted_states = val_episode(val_collect, epoch + 1, global_rngs)
        episode = test_dataset.get_episode(epoch + 1)

        if Long_Path_sim:
            ground_truth_x = x_list
            ground_truth_y =  y_list
            ground_truth_vx =  vx_list
            ground_truth_vy =  vy_list
            ground_truth_yawrate =  yawrate_list
        else:
            ground_truth_x = episode[:, 0]
            ground_truth_y =  episode[:, 1]
            ground_truth_vx =  episode[:, 3]
            ground_truth_vy =  episode[:, 4]
            ground_truth_yawrate =  episode[:, 5]

        # calculate value rmse
        position_error_rmse = compute_vertor_rmse(episode[:, 0][-prediction_length:], episode[:, 1][-prediction_length:], predicted_states[0, :, 0], predicted_states[0, :, 1])
        v_error_rmse = compute_vertor_rmse(episode[:, 3][-prediction_length:],episode[:, 4][-prediction_length:], predicted_states[0, :, 3], predicted_states[0, :, 4])
        yaw_error_rmse = compute_1d_rmse(episode[:, 5][-prediction_length:], predicted_states[0, :, 5])

        rmse_dict["position_error"].append(position_error_rmse)
        rmse_dict["v_error"].append(v_error_rmse)
        rmse_dict["yaw_rate_error"].append(yaw_error_rmse)

        if Save_Fig:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].plot(ground_truth_x, ground_truth_y, label='Ground Truth', marker='o', markersize=5)
            axs[0, 0].plot(predicted_states[0, :, 0], predicted_states[0, :, 1], label='Predicted', marker='x', markersize=5)
            axs[0, 0].legend()
            axs[0, 0].axis('equal')

            predict_x = np.arange(0, predicted_states.shape[1]) + episode.shape[0] - predicted_states.shape[1]
            axs[0, 1].plot(ground_truth_vx, label='Ground Truth vx')
            axs[0, 1].plot(ground_truth_vy, label='Ground Truth vy')
            axs[0, 1].plot(predict_x, predicted_states[0, :, 3], label='Predicted vx')
            axs[0, 1].plot(predict_x, predicted_states[0, :, 4], label='Predicted vy')
            axs[0, 1].legend()

            # labels = ['posi_err', 'v_err', 'yaw_rate_error']
            # rmse_data = [position_error_rmse, v_error_rmse, yaw_error_rmse]
            # axs[1,0].bar(labels, rmse_data)
            axs[1, 0].plot(episode[:, 2] * 57.3, label='Ground Truth yaw(deg)')
            axs[1, 0].plot(predict_x, predicted_states[0, :, 2] * 57.3, label='Predicted yaw(deg)')
            axs[1, 0].legend()

            axs[1, 1].plot(ground_truth_yawrate * 57.3, label='Ground Truth yawrate(deg)')
            axs[1, 1].plot(predict_x, predicted_states[0, :, 5] * 57.3, label='Predicted yawrate(deg)')
            axs[1, 1].plot(episode[:, 7] * 57.3 * 0.1, label='steer angle * 0.1(deg)')
            axs[1, 1].legend()

            fig.tight_layout()
            plt.title("prediction result data_" + str(epoch+1))
            save_path = os.path.join(fig_result_path, "result" + str(epoch+1))
            fig.savefig(save_path, format="png")
            plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    axs[0].scatter(range(len(rmse_dict["position_error"])), rmse_dict["position_error"], marker='o')
    axs[0].set_title("position_error rmse")

    axs[1].scatter(range(len(rmse_dict["v_error"])), rmse_dict["v_error"], marker='o')
    axs[1].set_title("v_error rmse")

    axs[2].scatter(range(len(rmse_dict["yaw_rate_error"])), rmse_dict["yaw_rate_error"], marker='o')
    axs[2].set_title("yaw_rate_error rmse")

    mean_rmse = [np.mean(rmse_dict["position_error"]),np.mean(rmse_dict["v_error"]),np.mean(rmse_dict["yaw_rate_error"])]
    print("position_error = " + str(np.mean(rmse_dict["position_error"])))
    print("v_error = " + str(np.mean(rmse_dict["v_error"])))
    print("yaw_rate_error = " + str(np.mean(rmse_dict["yaw_rate_error"])))

    axs[3].barh(list(rmse_dict.keys()), mean_rmse, color='skyblue')
    axs[3].set_title("mean RMSE fron all verify data")
    axs[3].set_xlabel("Values")

    fig.tight_layout()
    save_path = os.path.join(fig_result_path, "final result")
    fig.savefig(save_path, format="png")
    plt.show()

data_num = len(test_dataset)
# data_num = 100

rmse_dict = {
    "position_error":[],
    "v_error":[],
    "yaw_rate_error":[]
}

x_list = []
y_list = []
vx_list = []
vy_list = []
yawrate_list = []
if Long_Path_sim:
    for idx in range(data_num):
        if len(x_list)== 0:
            x_list = test_dataset.get_episode(idx)[:, 0].tolist()
            y_list = test_dataset.get_episode(idx)[:, 1].tolist()
            vx_list = test_dataset.get_episode(idx)[:, 3].tolist()
            vy_list = test_dataset.get_episode(idx)[:, 4].tolist()
            yawrate_list = test_dataset.get_episode(idx)[:, 5].tolist()
        else:
            x_list.extend(test_dataset.get_episode(idx)[:, 0][-10:].tolist())
            y_list.extend(test_dataset.get_episode(idx)[:, 1][-10:].tolist())
            vx_list.extend(test_dataset.get_episode(idx)[:, 3][-10:].tolist())
            vy_list.extend(test_dataset.get_episode(idx)[:, 4][-10:].tolist())
            yawrate_list.extend(test_dataset.get_episode(idx)[:, 5][-10:].tolist())

clear_directory(fig_result_path + "/")
plot_final_result(data_num, rmse_dict)
