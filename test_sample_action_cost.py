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
# from car_foundation.jax_models import JaxTransformerDecoder
from car_foundation.models import TorchMLP, TorchTransformer, TorchTransformerDecoder, TorchGPT2

from verify_utils import *
from sample_3b_calculate_func_param import Vehicle_param, get_matrix, lqr, calculate_feedback_coeff_list, calculate_kappa_coeff
import torch.optim as optim

Save_Fig = True

model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/test_compress_file/torch_model_400_use_cnn_before_emb_fine_tune_c2_more_data"
# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-03-11T19:25:14.129-model_checkpoint"
# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-03-20T19:50:30.419-model_checkpoint"
# dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/temp_debug_data'  #10 pkl
# dataset_path = '/disk1/collect_data_from_anycar/data_from_bag/new_temp_data/pkg_file'  #10 pkl
dataset_path = '/disk1/collect_data_from_anycar/select_bag_for_verify_sample/data_files'
# dataset_path = '/disk1/collect_data_from_anycar/data_from_bag/new_temp_data/bug_pkl'


fig_result_path = '/home/gzh/anycar/model_test_result_fig'
os.makedirs(fig_result_path, exist_ok=True)

use_torch_decoder = True

model_checkpint = 400

history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False
binary_mask = False
ATTACK = False  # verify data will not add noise


actual_input_length = 42

# 降采样
Compress_Sample = False
compress_history_length = 125
compress_ratio = compress_history_length / (history_length-1)
# 时序降采样
# unique_indexs = np.arange(0, history_length -1 , math.ceil(1 / compress_ratio))
# 缩短时序
unique_indexs = np.arange(history_length- compress_history_length, history_length, 1)

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1
USE_ZERO_POINT=True

dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
test_dataset = MujocoDataset(dataset_files, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT, with_ref_trajectory=True)

# Device to use
device = torch.device("cuda")
assert device.type == "cuda", "Only cuda is supported"

if use_torch_decoder:
    if Compress_Sample:
        actual_input_length = int(actual_input_length * compress_ratio)
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout, compress_history_length, prediction_length, actual_input_length)
    else:
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
    with torch.no_grad(): 
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
            last_pose = y_pred[:, i, :6]

        return y_pred

def val_episode(model, episode_num):
    with torch.no_grad():
        episode = test_dataset.get_episode(episode_num)

        episode = torch.unsqueeze(episode, 0)
        batch = episode[:, :, :-1]
        history, action, y, action_padding_mask = test_dataset[episode_num:episode_num+1]
        if Compress_Sample:
            history = history[:, unique_indexs, :]
        predicted_states = apply_batch_torch(model, batch[:, history_length-1, :], history, action, y, input_mean, input_std)
        predicted_states = predicted_states.cpu().detach().numpy()
        return np.array(predicted_states)

def compute_cost_value(ref_trajectory, predicted_states, current_state, q_list, r_list, cost_weight_for_delta_u):
    total_cost = 0.0
    previous_action = None

    # get action list
    vehicle_param = Vehicle_param()
    select_speed = 20.0
    vehicle_param.total_weight = 30000
    Q = np.array([0.05, 1.5, 0.3, 0.3, 0.03, 0, 0])
    R = 1
    backlash_value = 3.0 / 57.3 # 3 degree backlash value
    matrix_A, matrix_B = get_matrix(vehicle_param, select_speed)
    K, _, _ = lqr(matrix_A, matrix_B, np.diag(Q), R)

    # sample params
    Prediction_length_ = 50
    Ts_ = 0.05

    coeff_feedback = calculate_feedback_coeff_list(matrix_A, matrix_B, K, Ts_, Prediction_length_)
    feedforward_coeff = calculate_kappa_coeff(vehicle_param, select_speed)

    last_steer_angle = current_state[0,0]
    last_ff_planing_cmd = current_state[0,1]
    last_error_state = current_state[0,2:9]

    sample_action_list = np.full((50, 1), last_steer_angle)

    sample_action_list += np.dot(coeff_feedback, last_error_state) * Ts_
    sample_action_list += np.full((50, 1), last_ff_planing_cmd * feedforward_coeff)

    columns_to_extract = [0, 1, 4, 5, 6, 7]

    for t in range(predicted_states.shape[1]):
        # State error
        state_error = ref_trajectory[0,t,columns_to_extract] - predicted_states[0, t, :6]
        state_cost = np.sum(q_list * (state_error ** 2))

        # Action cost
        # action = episode[-prediction_length + t, 6:8]
        # action_cost = np.sum(r_list * (action ** 2))

        # Delta action cost
        # if previous_action is not None:
        #     delta_u = action - previous_action
        #     delta_u_cost = cost_weight_for_delta_u * np.sum(delta_u ** 2)
        # else:
        #     delta_u_cost = 0.0

        # previous_action = action

        # Accumulate total cost
        total_cost += state_cost # + action_cost + delta_u_cost

    return total_cost

# plot final result
def plot_final_result(data_num, rmse_dict):

    for epoch in range(data_num):
        
        if epoch + 1 >= len(test_dataset):
            continue
        
        predicted_states = val_episode(model, epoch + 1)
        episode = test_dataset.get_episode(epoch + 1)
        ref_trajectory = test_dataset.get_current_frame_state(epoch + 1)
        current_state =  test_dataset.get_current_state(epoch + 1)

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

        # calculate cost value to evaluate action list
        q_list = np.array([2.0, 0.05, 0.05, 0.0, 0.0, 0.05])
        r_list = np.array([1.0, 1.0])
        cost_weight_for_delta_u = 0.0

        cost_value = compute_cost_value(ref_trajectory, predicted_states,current_state, q_list, r_list, cost_weight_for_delta_u)

        if Save_Fig:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].plot(ground_truth_x[230:], ground_truth_y[230:], label='Ground Truth', marker='o', markersize=5)
            axs[0, 0].plot(predicted_states[0, :, 0], predicted_states[0, :, 1], label='Predicted', marker='x', markersize=5)
            axs[0, 0].plot(ref_trajectory[0,:,0], ref_trajectory[0,:,1], label='Ref trajectory', marker='o', markersize=5)
            axs[0, 0].legend()
            axs[0, 0].axis('equal')

            predict_x = np.arange(0, predicted_states.shape[1]) + episode.shape[0] - predicted_states.shape[1]
            axs[0, 1].plot(predict_x, ground_truth_vx[-50:], label='Ground Truth vx')
            # axs[0, 1].plot(ground_truth_vy, label='Ground Truth vy')
            axs[0, 1].plot(predict_x, predicted_states[0, :, 3], label='Predicted vx')
            axs[0, 1].plot(predict_x, ref_trajectory[0,:,6], label='Ref trajectory')
            # axs[0, 1].plot(predict_x, predicted_states[0, :, 4], label='Predicted vy')
            axs[0, 1].legend()

            # labels = ['posi_err', 'v_err', 'yaw_rate_error']
            # rmse_data = [position_error_rmse, v_error_rmse, yaw_error_rmse]
            # axs[1,0].bar(labels, rmse_data)
            axs[1, 0].plot(predict_x, episode[:, 2][-50:] * 57.3, label='Ground Truth yaw(deg)')
            axs[1, 0].plot(predict_x, predicted_states[0, :, 2] * 57.3, label='Predicted yaw(deg)')
            axs[1, 0].plot(predict_x, ref_trajectory[0,:,4] * 57.3, label='reference yaw(deg)')
            axs[1, 0].legend()

            axs[1, 1].plot(predict_x, ground_truth_yawrate[-50:] * 57.3, label='Ground Truth yawrate(deg)')
            axs[1, 1].plot(predict_x, predicted_states[0, :, 5] * 57.3, label='Predicted yawrate(deg)')
            axs[1, 1].plot(predict_x, ref_trajectory[0,:,5] * 57.3, label='reference yawrate(deg)')
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
