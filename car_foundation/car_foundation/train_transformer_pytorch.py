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

history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False

ATTACK = False
USE_ZERO_POINT= True

lr_begin = 5e-4
warmup_period = 500
num_epochs = 400
load_checkpoint = False
resume_model_checkpint = 0
resume_model_name = ""

val_every = 50
batch_size = 512
lambda_l2 = 1e-4
#dataset_path = 'DATASET-PATH'
dataset_path = '/disk1/collect_data_from_anycar/New_demo/new_data_with_x_mean_zero/total_data_1'
# dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/temp_debug_data'
# check_data_path = '/disk1/collect_data_from_anycar/temp_verify_backlash_model/2025-01-14T18:15:29.673-nuplan-dynamic-model-verify'
check_data_path = '/disk1/collect_data_from_anycar/New_demo/check_data_with_offset'
comment = 'jax'

# Device to use
device = torch.device("cuda")
assert device.type == "cuda", "Only cuda is supported"

resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}")

num_workers = 6

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1

# state_dim = 6
# action_dim = 2
# latent_dim = 64
# num_heads = 4
# num_layers = 2
# dropout = 0.1

save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
save_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')

# architecture = 'decoder'
# architecture = 'mlp'
# architecture = 'cnn'
# architecture = 'torch'
architecture = 'torch_decoder'

if architecture == 'decoder':
    model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length - 1, prediction_length, jnp.bfloat16, name=architecture)
elif architecture == 'mlp':
    model = JaxMLP([256, 256, 256, 256, 256], state_dim, 0.1, name=architecture)
elif architecture == 'cnn':
    model = JaxCNN([32, 64, 128, 256], state_dim, 0.1, name=architecture)
elif architecture == 'torch':
    model = TorchTransformer(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout)
elif architecture == "torch_decoder":
    model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout).to(device) 

# Load the dataset
binary_mask = False # type(model) == TorchGPT2
dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
random.shuffle(dataset_files)
total_len = len(dataset_files)
split_70 = int(total_len * 0.7)
split_20 = int(total_len * 0.9)
data_70 = dataset_files[:split_70]
data_20 = dataset_files[split_70:split_20]
data_10 = dataset_files[split_20:]

train_dataset = MujocoDataset(data_70, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT)
# import ipdb; ipdb.set_trace()
print("train data length", len(train_dataset))

val_dataset = MujocoDataset(data_20, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK, use_zero_point=USE_ZERO_POINT)
test_dataset = MujocoDataset(data_10, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK, use_zero_point=USE_ZERO_POINT)

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
        "resume_checkpoint_path": resume_model_folder_path,
        "resume_checkpoint": resume_model_checkpint,
        "attack": ATTACK,
    }
)
print(wandb.config)
print(f"total params: {sum(p.numel() for p in model.parameters())}")

def create_learning_rate_fn():
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=lr_begin, transition_steps=warmup_period)
    decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_period]
    )
    return schedule_fn

learning_rate_fn = create_learning_rate_fn()

model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr_begin, weight_decay=lambda_l2)

input_mean = torch.tensor(train_dataset.mean, dtype=torch.float32)
input_std = torch.tensor(train_dataset.std, dtype=torch.float32)
input_mean = input_mean.to(device)
input_std = input_std.to(device)


print("mean: ", input_mean.tolist())
print("std: ", input_std.tolist())

# def apply_batch(var_collect, last_state, history, action, y, action_padding_mask, rngs):
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

# def val_episode(var_collect, episode_num, rngs, dateset):
#     episode = dateset.get_episode(episode_num)
#     episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
#     batch = episode[:, :, :-1]
#     history, action, y, action_padding_mask = dateset[episode_num:episode_num+1]
#     history = jnp.array(history.numpy())
#     action = jnp.array(action.numpy())
#     y = jnp.array(y.numpy())
#     action_padding_mask = jnp.array(action_padding_mask.numpy())
#     predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs)
#     return np.array(predicted_states)

# def visualize_episode(epoch_num: int, episode_num, val_dataset, rngs):
#     val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     raw_restored = orbax_checkpointer.restore(os.path.join(save_model_folder_path, f"{epoch_num}", "default"))
#     # import ipdb; ipdb.set_trace()
#     val_collect['params'] = raw_restored['model']['params']
#     predicted_states = val_episode(val_collect, episode_num, rngs, val_dataset)
#     episode = val_dataset.get_episode(episode_num)

#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#     axs[0, 0].plot(episode[:, 0], episode[:, 1], label='Ground Truth', marker='o', markersize=5)
#     axs[0, 0].plot(predicted_states[0, :, 0], predicted_states[0, :, 1], label='Predicted', marker='x', markersize=5)
#     axs[0, 0].legend()
#     axs[0, 0].axis('equal')

#     predict_x = np.arange(0, predicted_states.shape[1]) + episode.shape[0] - predicted_states.shape[1]
#     axs[0, 1].plot(episode[:, 3], label='Ground Truth vx')
#     axs[0, 1].plot(episode[:, 4], label='Ground Truth vy')
#     axs[0, 1].plot(predict_x, predicted_states[0, :, 3], label='Predicted vx')
#     axs[0, 1].plot(predict_x, predicted_states[0, :, 4], label='Predicted vy')
#     axs[0, 1].legend()

#     axs[1, 1].plot(episode[:, 5], label='Ground Truth v_yaw')
#     axs[1, 1].plot(predict_x, predicted_states[0, :, 5], label='Predicted v_yaw')
#     axs[1, 1].legend()

#     fig.tight_layout()
#     fig.savefig('episode.png')
#     plt.close(fig)
#     wandb.log({"episode": wandb.Image('episode.png')})

train_losses = []
val_losses = []
val_epoch_nums = []

# for epoch in range(num_epochs):
#     running_loss = 0.0
#     t = tqdm.tqdm(train_loader)

#     running_loss /= len(train_loader)
#     train_losses.append(running_loss)
#     wandb.log({"train_loss": running_loss, "learning_rate": learning_rate_fn(global_state.step)})
#     print(save_model_folder_path)
#     # import ipdb; ipdb.set_trace()
#     ckpt = {'model': global_state, 'input_mean': input_mean, 'input_std': input_std}
#     save_args = orbax_utils.save_args_from_target(ckpt)
#     checkpoint_manager.save(epoch+1, ckpt, save_kwargs={'save_args': save_args})

    # if (epoch + 1) % val_every == 0:
    #     visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
    #     val_loss = val_loop(global_state, global_var, val_loader, global_rngs)
    #     val_losses.append(val_loss)
    #     val_epoch_nums.append(epoch + 1)
    #     print(f'Validation Loss: {val_loss:.4f}')
    #     wandb.log({"val_loss": val_loss})

    # if (epoch+1) % num_epochs == 0:
    #     temp_use_verify_data()

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

    # print(f"原始 action 的维度: {action.shape}")
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
        optimizer.zero_grad(set_to_none=True)

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

        train_loss += loss

    # if (epoch + 1) % val_every == 0:
    #     model.eval()
    #     with torch.no_grad():
    #         test_loss = 0.
    #         if test_loader is not None:
    #             for inputs, targets in test_loader:
    #                 inputs = inputs.to(device)
    #                 targets = targets.to(device)
    #                 outputs = model.predict(inputs)
    #                 # test_loss += criterion(outputs, targets).item()
    #                 test_loss = torch.mean((outputs - targets) ** 2, dim=0) + test_loss
                
    #                 test_loss /= len(test_loader)
            
    #         train_loss = 0.
    #         for inputs, targets in train_loader:
    #             inputs = inputs.to(device)
    #             targets = targets.to(device)
    #             outputs = model(inputs)
    #             train_loss += criterion(outputs, targets).item()
    #         train_loss /= len(train_loader)
        if (epoch + 1) % val_every == 0:
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

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

            torch.save(checkpoint, save_model_folder_path)


    #     val_losses.append(test_loss)
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    log_dict = {"train_loss": train_loss}
    # log_dict.update({f"test_loss_{i}": test_loss[i] for i in range(test_loss.shape[0])})
    print("train_loss = " + str(train_loss))
    # print(f"Epoch {epoch}, Validation Loss: {valid_loss}")

torch.save(model.state_dict(), save_model_folder_path)

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

torch.save(checkpoint, save_model_folder_path)

train_epoch_nums = list(range(1, num_epochs + 1))
plt.figure()
train_losses_np = [loss.cpu().detach().numpy() for loss in train_losses]
plt.plot(train_epoch_nums, train_losses_np, label='Train Loss')
plt.plot(val_epoch_nums, val_losses, label='Val Loss')
plt.legend()
plt.savefig('train_val_loss.png')
# # plt.show()

# # model.eval()
# visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
# test_loss = val_loop(global_state, global_var, test_loader, global_rngs)
# print(f'Test Loss: {test_loss:.4f}')

# # Save the model
# # torch.save(model.state_dict(), 'model.pth')
# wandb.save('model_checkpoint/')

wandb.finish()
