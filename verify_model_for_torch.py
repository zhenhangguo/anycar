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
from car_foundation.models import TorchMLP, TorchTransformer, TorchTransformerDecoder, TorchGPT2

from verify_utils import *
import torch.optim as optim

Save_Fig = True

# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-02-10T10:20:21.689-model_checkpoint"
model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-03-11T19:25:14.129-model_checkpoint"
dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/temp_debug_data'  #10 pkl
# dataset_path = '/disk1/collect_data_from_anycar/check_data/verify_bag_data_0310'  #10 pkl

fig_result_path = '/home/gzh//anycar/model_test_result_fig'

use_torch_decoder = True

model_checkpint = 400

history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False
binary_mask = False
ATTACK = False  # verify data will not add noise

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1
USE_ZERO_POINT=True

dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
test_dataset = MujocoDataset(dataset_files, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT)

# Device to use
device = torch.device("cuda")
assert device.type == "cuda", "Only cuda is supported"

if use_torch_decoder:
    model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout)
else:
    model = TorchTransformer(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout)
model = model.to(device)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

optimizer = optim.AdamW(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# set mean and std for checkpoint or other data
input_mean = torch.tensor(checkpoint['input_mean'], dtype=torch.float32)
input_std = torch.tensor(checkpoint['input_std'], dtype=torch.float32)
input_mean = input_mean.to(device)
input_std = input_std.to(device)

print("input_mean = " + str(input_mean))
print("input_std = " + str(input_std))

print("model_hash = " + str(model_hash(model.state_dict())))

def align_yaw_torch(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))
    return d_yaw_aligned + yaw_2

def apply_batch_torch(model, last_state, history, action, y, input_mean, input_std):
    history = history.to(device)
    y = y.to(device)

    if use_torch_decoder:
        history[:, :, :6] = (history[:, :, :6] - input_mean) / input_std
    else:
        history = (history[:, :, :6] - input_mean) / input_std
    y[:, :, :6] = (y[:, :, :6] - input_mean) / input_std

    x = history[:, 1:, :]
    x = x.to(device)
    action = action.to(device)

    y_pred = model(x, action) * input_std + input_mean
    last_pose = last_state[:, :6].to(device)
    for i in range(y_pred.shape[1]):
        # rotate dx, dy back to world frame
        y_pred_x = y_pred[:, i, 0] * torch.cos(last_pose[:, 2]) - y_pred[:, i, 1] * torch.sin(last_pose[:, 2])
        y_pred_y = y_pred[:, i, 0] * torch.sin(last_pose[:, 2]) + y_pred[:, i, 1] * torch.cos(last_pose[:, 2])
        y_pred[:, i, 0] = y_pred_x
        y_pred[:, i, 1] = y_pred_y
        # accumulate the poses
        y_pred[:, i, :6] += last_pose
        y_pred[:, i, 2] = align_yaw_torch(y_pred[:, i, 2], 0.0)
        last_pose = y_pred[:, i, :6]
    return y_pred

def val_episode(model, episode_num):
    episode = test_dataset.get_episode(episode_num)
    episode = torch.unsqueeze(episode, 0)
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask = test_dataset[episode_num:episode_num+1]
    predicted_states = apply_batch_torch(model, batch[:, history_length-1, :], history, action, y, input_mean, input_std)
    predicted_states = predicted_states.cpu().detach().numpy()
    return np.array(predicted_states)

# plot final result
def plot_final_result(data_num, rmse_dict):

    for epoch in range(data_num):
        
        if epoch + 1 >= len(test_dataset):
            continue
        
        predicted_states = val_episode(model, epoch + 1)
        episode = test_dataset.get_episode(epoch + 1)

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

clear_directory(fig_result_path + "/")
plot_final_result(data_num, rmse_dict)
