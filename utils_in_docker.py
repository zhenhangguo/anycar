import hashlib
# from jax.tree_util import tree_flatten
# import torch
import os
import glob

# import jax.numpy as jnp
import numpy as np
# from car_foundation.utils import align_yaw_jax
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
# from torch.utils.data import DataLoader


# def model_hash(params):
#     md5 = hashlib.md5()
#     flat_params, _ = tree_flatten(params)
#     for param in flat_params:
#         md5.update(jnp.array(param).tobytes())
#     return md5.hexdigest()

# def compute_1d_rmse(tensor1, tensor2):
#     assert tensor1.shape == tensor2.shape
#     mse = torch.mean((tensor1 - tensor2) ** 2)
#     rmse = torch.sqrt(mse).item()
#     return rmse

# def compute_vertor_rmse(tensor1_x, tensor1_y, tensor2_x, tensor2_y):
#     assert tensor1_x.shape == tensor2_x.shape
#     assert tensor1_y.shape == tensor2_y.shape
#     x_error = tensor1_x - tensor2_x
#     y_error = tensor1_y - tensor2_y
#     position_error = x_error ** 2 + y_error **2
#     mse = torch.mean(position_error)
#     rmse = torch.sqrt(mse).item()
#     return rmse

def clear_directory(path):
    files = glob.glob(os.path.join(path, '*'))
    
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"无法删除 {f}: {e}")

# def apply_batch(var_collect, last_state, history, action, y, action_padding_mask, rngs, input_mean, input_std, model):
#     history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
#     y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)

#     x = history[:, 1:, :]
#     # tgt_mask = nn.Transformer.generate_square_subsequent_mask(action.size(1), device=action.device)
#     y_pred = model.apply(var_collect, x, action, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=True) * input_std + input_mean
#     last_pose = last_state[:, :6]
#     for i in range(y_pred.shape[1]):
#         # rotate dx, dy back to world frame
#         y_pred_x = y_pred[:, i, 0] * jnp.cos(last_pose[:, 2]) - y_pred[:, i, 1] * jnp.sin(last_pose[:, 2])
#         y_pred_y = y_pred[:, i, 0] * jnp.sin(last_pose[:, 2]) + y_pred[:, i, 1] * jnp.cos(last_pose[:, 2])
#         y_pred = y_pred.at[:, i, 0].set(y_pred_x)
#         y_pred = y_pred.at[:, i, 1].set(y_pred_y)
#         # accumulate the poses
#         y_pred = y_pred.at[:, i, :6].add(last_pose)
#         y_pred = y_pred.at[:, i, 2].set(align_yaw_jax(y_pred[:, i, 2], 0.0))
#         last_pose = y_pred[:, i, :6]
#     return y_pred

def show_picture(data_analysis_mean, data_analysis_std):
    _, axes = plt.subplots(7, 1, figsize=(6, 18))
    axes = axes.flatten()

    pass_id_key = 0
    for i, (key, value) in enumerate(data_analysis_mean.items()):
        if key == "id":
            pass_id_key = -1
            continue
        ax = axes[i + pass_id_key]
        ax.scatter(range(len(value)), value, label=f"{key}")
        ax.set_title(key)

    plt.tight_layout()

    _, axes = plt.subplots(7, 1, figsize=(6, 18))
    axes = axes.flatten()

    pass_id_key = 0
    for i, (key, value) in enumerate(data_analysis_std.items()):
        if key == "id":
            pass_id_key = -1
            continue
        ax = axes[i + pass_id_key]
        ax.scatter(range(len(value)), value, label=f"{key}")
        ax.set_title(key)

    plt.tight_layout()
    plt.show()

def calculate_data_range(test_dataset, data_analysis_mean, data_analysis_std):
    id = 0
    for history, action, y, action_padding_mask in test_dataset:

        data_analysis_mean["x_mean"].append(history[:,0].mean())
        data_analysis_std["x_std"].append(history[:,0].std())

        data_analysis_mean["y_mean"].append(history[:,1].mean())
        data_analysis_std["y_std"].append(history[:,1].std())

        data_analysis_mean["yaw_mean"].append(history[:,2].mean())
        data_analysis_std["yaw_std"].append(history[:,2].std())

        data_analysis_mean["vx_mean"].append(history[:,3].mean())
        data_analysis_std["vx_std"].append(history[:,3].std())

        data_analysis_mean["vy_mean"].append(history[:,4].mean())
        data_analysis_std["vy_std"].append(history[:,4].std())

        data_analysis_mean["yawrate_mean"].append(history[:,5].mean())
        data_analysis_std["yawrate_std"].append(history[:,5].std())

        data_analysis_mean["steer_mean"].append(history[:,7].mean())
        data_analysis_std["steer_std"].append(history[:,7].std())
        
        data_analysis_mean["id"].append(id)
        data_analysis_std["id"].append(id)
        
        id = id + 1

def plot_key():
    key_list = ["pos_x", "pos_y", "yaw", "vx", "vy", "yawrate", "steer_angle"]
    return key_list

def plot_data_density(test_dataset):

    data_total = test_dataset.get_total_data()
    key_list = plot_key()

    _, axes = plt.subplots(7, 1, figsize=(10, 18))
    axes = axes.flatten()

    def plot_sub_fig(axe, title, data):
        sns.histplot(data, bins=30, kde=True, ax=axe, color='blue', edgecolor='black', alpha=0.6)
        vals = axe.get_yticks()
        axe.set_yticklabels(['{:.0f}%'.format(val * 100 / len(data)) for val in vals])
        axe.set_xlabel('Value')
        axe.set_ylabel('Density/Frequency%')
        axe.grid(axis='y', linestyle='--', alpha=0.7)
        axe.set_title(title)

    for id in range(len(key_list)):
        if key_list[id] == "steer_angle":
            array = data_total[:, :, id + 1].numpy().flatten()
        else:
            array = data_total[:, :, id].numpy().flatten()
        plot_sub_fig(axes[id], key_list[id], array)

    plt.tight_layout()
    plt.show()


def get_dict_mean_and_std():
    data_analysis_mean = {
    "x_mean": [],
    "y_mean": [],
    "yaw_mean": [],
    "vx_mean": [],
    "vy_mean": [],
    "yawrate_mean": [],
    "steer_mean": [],
    "id" : [],
    }

    data_analysis_std = {
        "x_std": [],
        "y_std": [],
        "yaw_std": [],
        "vx_std": [],
        "vy_std": [],
        "yawrate_std" : [],
        "steer_std" : [],  
        "id" : []  ,
    }
    
    return data_analysis_mean, data_analysis_std

def quaternion_to_euler(q):
    # Normalize quaternion
    norm = np.linalg.norm(q, axis=1)[:, np.newaxis]
    q = q / norm
    
    # Extract the values from Q
    q_w, q_x, q_y, q_z = q[:,0], q[:,1], q[:,2], q[:,3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x**2 + q_y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sinp) >= 1,
                    np.sign(sinp) * np.pi / 2,  # use 90 degrees if out of range
                    np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y**2 + q_z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # avoid yaw list jump because of angle limit
    yaw = np.unwrap(yaw, discont=np.pi)

    return roll, pitch, yaw

def clear_unfit_pkl_file(dataset_path):
    pickle_files = glob.glob(os.path.join(dataset_path, '*.pkl'))
    
    file_data_pairs = []

    for file_path in pickle_files:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            file_data_pairs.append((file_path, data))
    
    # set data feature bound and remove this pkl file
    max_vx_speed = 30
    min_vx_speed = 5

    key_list = ["steer", "xpos_x", "xpos_y", "xpos_z", "xori_x", "xori_y", "xori_z", "xori_w", "avel_z", "xacc_x", "xacc_y", "xacc_z", "throttle", "xvel_x"]
    
    for file_path, data in file_data_pairs:
        # remove data if speed too high or too low
        if (max(data.data_logs["xvel_x"]) > max_vx_speed) or (min(data.data_logs["xvel_x"]) < min_vx_speed):
            os.remove(file_path)
        
        # remove data if any key in key_list has NaN values
        for key in key_list:
            if np.any(np.isnan(data.data_logs[key])):
                os.remove(file_path)
                break
        
        # remove data if yaw angle appera jump
        q = np.array([data.data_logs["xori_w"],data.data_logs["xori_x"], data.data_logs["xori_y"], data.data_logs["xori_z"]]).T
        _, _, yaw = quaternion_to_euler(q)
        if np.max(np.abs(yaw[1:] - yaw[:-1])) > 0.017452007:
            np.set_printoptions(threshold=np.inf)
            print("yaw jump")
            print(np.array(yaw))
            os.remove(file_path)

def calculate_dataset_md5(dataset):
    """
    计算 PyTorch Dataset 的 MD5 值
    参数:
    - dataset: torch.utils.data.Dataset 对象
    返回:
    - MD5 值
    """
    hash_md5 = hashlib.md5()

    for data in DataLoader(dataset, batch_size=1, shuffle=False):  # 保证数据顺序一致
        serialized_data = str(data).encode("utf-8")  # 将数据转为字符串再编码
        hash_md5.update(serialized_data)

    return hash_md5.hexdigest()