from functools import partial
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import numpy as np
import optax
import orbax

from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR
from car_foundation.dataset import DynamicsDataset, IssacSimDataset, MujocoDataset
from car_foundation.models import TorchMLP, TorchTransformer, TorchTransformerDecoder, TorchGPT2
from car_foundation.jax_models import JaxTransformerDecoder, JaxMLP, JaxCNN
from car_foundation.utils import generate_subsequences, generate_subsequences_hf, align_yaw, align_yaw_jax
import datetime
import os
import glob
import time
import math
import random
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pytorch_warmup as warmup
import torch.optim as optim
import hashlib
import sys

from rich.progress import track

PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"

torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

history_length = 250
actual_input_length = 42

prediction_length = 50
delays = None
teacher_forcing = False

# 降采样
Compress_Sample = False
compress_history_length = 125
compress_ratio = compress_history_length / history_length
# 时序降采样
# unique_indexs = np.arange(0, history_length -1 , math.ceil(1 / compress_ratio))
# 缩短时序
unique_indexs = np.arange(history_length- compress_history_length, history_length, 1)

ATTACK = False
USE_ZERO_POINT= True

FINE_TUNE = False

if FINE_TUNE:
    lr_begin = 5e-4
    warmup_period = 2
    num_epochs = 400
    load_checkpoint = True
    resume_model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/test_compress_file/torch_model_240_use_cnn_before_emb"
else:
    lr_begin = 5e-4
    warmup_period = 500
    num_epochs = 400
    load_checkpoint = False

val_every = 20
batch_size = 512
lambda_l2 = 1e-4
# dataset_path = '/disk1/collect_data_from_anycar/data_from_bag/new_temp_data/pkg_file'

dataset_path = '/disk1/collect_data_from_anycar/New_demo/new_data_with_x_mean_zero/total_data_1'
# dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/temp_debug_data'
# dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-01-14T16:39:46.443-nuplan-dynamic-model-base'
# check_data_path = '/disk1/collect_data_from_anycar/temp_verify_backlash_model/2025-01-14T18:15:29.673-nuplan-dynamic-model-verify'
check_data_path = '/disk1/collect_data_from_anycar/New_demo/check_data_with_offset'
comment = 'torch'

# Device to use
device = torch.device("cuda")
assert device.type == "cuda", "Only cuda is supported"

num_workers = 6

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1

save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
save_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')

architecture = 'torch_decoder'
use_torch_decoder = True

if architecture == "torch_decoder":
    if Compress_Sample:
        actual_input_length = int(actual_input_length * compress_ratio)
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout, compress_history_length, prediction_length, actual_input_length).to(device) 
    else:
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout).to(device) 

# Load the dataset
binary_mask = False
dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
random.shuffle(dataset_files)
total_len = len(dataset_files)
split_70 = int(total_len * 0.7)
split_20 = int(total_len * 0.9)
data_70 = dataset_files[:split_70]
data_20 = dataset_files[split_70:split_20]
data_10 = dataset_files[split_20:]

train_dataset = MujocoDataset(data_70, history_length+1, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT)
print("train data length", len(train_dataset))

val_dataset = MujocoDataset(data_20, history_length+1, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK, use_zero_point=USE_ZERO_POINT)
test_dataset = MujocoDataset(data_10, history_length+1, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK, use_zero_point=USE_ZERO_POINT)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

num_steps_per_epoch = len(train_loader)

wandb.init(
    # set the wandb project where this run will be logged
    project="transformer-sequence-prediction",
    name=architecture,

    # track hyperparameters and run metadata
    config={
        "history_length": history_length,
        "prediction_length": prediction_length,
        "delays": delays,
        "teacher_forcing": teacher_forcing,

        "learning_rate": lr_begin,
        "warmup_period": warmup_period,
        "architecture": architecture,
        "dataset": "even_dist_data",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lambda_l2": lambda_l2,
        "dataset_path": dataset_path.split('/')[-1],
        "comment": comment,

        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "implementation": "torch",
        "model_path": save_model_folder_path,
        "resume": load_checkpoint,
        "attack": ATTACK,
    }
)
print(wandb.config)
print(f"total params: {sum(p.numel() for p in model.parameters())}")

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

        if Compress_Sample:
            x = x[:, unique_indexs, :]

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
def val_episode(model, episode_num, dateset):
    episode = dateset.get_episode(episode_num)
    episode = torch.unsqueeze(episode, 0)
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask = dateset[episode_num:episode_num+1]
    predicted_states = apply_batch_torch(model, batch[:, history_length-1, :], history, action, y, input_mean, input_std)
    return np.array(predicted_states.cpu())

def val_loop(model_path, val_loader):
    if Compress_Sample:
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout, compress_history_length, prediction_length, actual_input_length)
    else:
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    val_loss = 0.0
    t_val = tqdm.tqdm(val_loader)
    for i, (history, action, y, action_padding_mask) in enumerate(t_val):
        history = history.to(device)
        action = action.to(device)
        y = y.to(device)
        action_padding_mask = action_padding_mask.to(device)
        val_loss += loss_fn(model, history, action, y, action_padding_mask).detach().item()
        t_val.set_description(f'Validation Loss: {(val_loss / (i + 1)):.4f}')
        t_val.refresh()
    val_loss /= len(val_loader)
    return val_loss
def visualize_episode(episode_num, val_dataset, model_path):
    if Compress_Sample:
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout, compress_history_length, prediction_length, actual_input_length)
    else:
        model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout) 
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    with torch.no_grad():
        predicted_states = val_episode(model, episode_num, val_dataset)
        episode = val_dataset.get_episode(episode_num)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(episode[:, 0], episode[:, 1], label='Ground Truth', marker='o', markersize=5)
    axs[0, 0].plot(predicted_states[0, :, 0], predicted_states[0, :, 1], label='Predicted', marker='x', markersize=5)
    axs[0, 0].legend()
    axs[0, 0].axis('equal')

    predict_x = np.arange(0, predicted_states.shape[1]) + episode.shape[0] - predicted_states.shape[1]
    axs[0, 1].plot(episode[:, 3], label='Ground Truth vx')
    axs[0, 1].plot(episode[:, 4], label='Ground Truth vy')
    axs[0, 1].plot(predict_x, predicted_states[0, :, 3], label='Predicted vx')
    axs[0, 1].plot(predict_x, predicted_states[0, :, 4], label='Predicted vy')
    axs[0, 1].legend()

    axs[1, 1].plot(episode[:, 5], label='Ground Truth v_yaw')
    axs[1, 1].plot(predict_x, predicted_states[0, :, 5], label='Predicted v_yaw')
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig('episode.png')
    plt.close(fig)
    wandb.log({"episode": wandb.Image('episode.png')})

def create_learning_rate_fn():
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=lr_begin, transition_steps=warmup_period)
    decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_period]
    )
    return schedule_fn

learning_rate_fn = create_learning_rate_fn()

if FINE_TUNE:
    model.load_state_dict(torch.load(resume_model_path)['model_state_dict'])
    input_mean = torch.load(resume_model_path)['input_mean']
    input_std = torch.load(resume_model_path)['input_std']
else:
    input_mean = torch.tensor(train_dataset.mean, dtype=torch.float32)
    input_std = torch.tensor(train_dataset.std, dtype=torch.float32)

input_mean = input_mean.to(device)
input_std = input_std.to(device)

model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr_begin, weight_decay=lambda_l2)
print("mean: ", input_mean.tolist())
print("std: ", input_std.tolist())

train_losses = []
val_losses = []
val_epoch_nums = []
# loss function for torch
def loss_fn(model, history, action, y, action_padding_mask):
    if architecture == "torch_decoder":
        history[:, :, :6] = (history[:, :, :6] - input_mean) / input_std
    else:
        history = (history[:, :, :6] - input_mean) / input_std
    y = (y[:, :, :6] - input_mean) / input_std
    history = history[:, 1:, :].detach()  
    y = y.detach()
    action = action.detach()  

    if Compress_Sample:
        history = history[:, unique_indexs, :]

    y_pred = model(history, action, history_padding_mask=None, action_padding_mask=action_padding_mask)
    action_padding_mask_binary = (action_padding_mask == 0)[:, :, None]    
    # add different state weights
    state_weights = torch.tensor([0.5, 0.5, 2.0, 0.5, 0.0, 2.5], dtype=torch.float32, device=device)  # 每个状态的权重, X, Y, yaw, Vx, Vy, yawrate
    state_weights = state_weights[None, None, :]  # 调整形状为 (1, 1, 6)
    loss = torch.mean(((y_pred - y) ** 2) * action_padding_mask_binary * state_weights)

    return loss

train_losses = []
val_losses = []
global_step = 0

for epoch in track(range(num_epochs)):
    t = tqdm.tqdm(train_loader)
    model.train()
    train_loss = 0.0
    for i, (history, action, y, action_padding_mask) in enumerate(t):
        history = history.to(device)
        action = action.to(device)
        y = y.to(device)
        action_padding_mask = action_padding_mask.to(device)

        # 更新学习率
        global_step = epoch * len(train_loader) + i 
        lr = learning_rate_fn(global_step)
        lr = torch.tensor(float(lr), dtype=torch.float32)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss = loss_fn(model, history, action, y, action_padding_mask)
        # 反向传播
        loss.backward()
        optimizer.step()  

        train_loss += loss.detach().item()

        t.set_description(f'Epoch {epoch + 1}, Loss: {(train_loss / (i + 1)):.4f}, LR: {learning_rate_fn(global_step):.6f}')
        t.refresh()

        optimizer.zero_grad(set_to_none=True)
        loss.detach_()

    if (epoch + 1) % val_every == 0:

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'input_mean': input_mean,
            'input_std': input_std,
            'epoch': epoch
        }

        save_model_folder_path_epoch = os.path.join(save_model_folder_path, f"{epoch + 1}", f"torch_model_{epoch + 1}")
        os.makedirs(os.path.dirname(save_model_folder_path_epoch), exist_ok=True)
        torch.save(checkpoint, save_model_folder_path_epoch)

        visualize_episode(1, val_dataset, save_model_folder_path_epoch)
        val_loss = val_loop(save_model_folder_path_epoch, val_loader)
        val_losses.append(val_loss)
        val_epoch_nums.append(epoch + 1)
        print(f'Validation Loss: {val_loss:.4f}')
        wandb.log({"val_loss": val_loss})

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    log_dict = {"train_loss": train_loss}
    wandb.log({"train_loss": train_loss, "learning_rate": learning_rate_fn(global_step)})
    print("train_loss = " + str(train_loss))

train_epoch_nums = list(range(1, len(train_losses) + 1))
plt.figure()
plt.plot(train_epoch_nums, train_losses, label='Train Loss')
plt.plot(val_epoch_nums, val_losses, label='Val Loss')
plt.legend()
plt.savefig('train_val_loss.png')
# plt.show()

wandb.finish()
