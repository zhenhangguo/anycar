from functools import partial
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from flax.training import orbax_utils
# from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state

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
import hashlib
from jax.tree_util import tree_flatten
import sys

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

FINE_TUNE = False
ATTACK = False
USE_ZERO_POINT= True

Run_first = True

if FINE_TUNE:
    lr_begin = 5e-4
    warmup_period = 2
    num_epochs = 400
    load_checkpoint = True
    resume_model_checkpint = 200
    resume_model_name = "2025-01-15T10:59:30.973-model_checkpoint"
else:
    lr_begin = 5e-4
    warmup_period = 500
    num_epochs = 400
    load_checkpoint = False
    resume_model_checkpint = 0
    resume_model_name = ""

val_every = 20
batch_size = 512 #1024
lambda_l2 = 1e-4
#dataset_path = 'DATASET-PATH'
dataset_path = '/disk1/collect_data_from_anycar/New_demo/total_data_2'
check_data_path = '/disk1/collect_data_from_anycar/temp_verify_backlash_model/2025-01-14T18:15:29.673-nuplan-dynamic-model-verify'
comment = 'jax'


resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}")

num_workers = 6

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1

save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
save_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')

architecture = 'decoder'
# architecture = 'mlp'
# architecture = 'cnn'

# model = TorchTransformer(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout)

if architecture == 'decoder':
    model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length - 1, prediction_length, jnp.bfloat16, name=architecture)
elif architecture == 'mlp':
    model = JaxMLP([256, 256, 256, 256, 256], state_dim, 0.1, name=architecture)
elif architecture == 'cnn':
    model = JaxCNN([32, 64, 128, 256], state_dim, 0.1, name=architecture)


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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
        "implementation": "jax",
        "model_path": save_model_folder_path,
        "resume": load_checkpoint,
        "resume_checkpoint_path": resume_model_folder_path,
        "resume_checkpoint": resume_model_checkpint,
        "attack": ATTACK,
    }
)
print(wandb.config)
# print(f"total params: {sum(p.numel() for p in model.parameters())}")

rng = jax.random.PRNGKey(3407)
rng, params_rng = jax.random.split(rng)
rng, dropout_rng = jax.random.split(rng)
init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}
global_rngs = init_rngs

jax_history_input = jnp.ones((batch_size, history_length-1, state_dim + action_dim), dtype=jnp.float32)
jax_history_mask = jnp.ones((batch_size, (history_length-1) * 2 - 1), dtype=jnp.float32)
jax_prediction_input = jnp.ones((batch_size, prediction_length, action_dim), dtype=jnp.float32)
jax_prediction_mask = jnp.ones((batch_size, prediction_length), dtype=jnp.float32)

def create_learning_rate_fn():
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=lr_begin, transition_steps=warmup_period)
    decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_period]
    )
    return schedule_fn

learning_rate_fn = create_learning_rate_fn()

global_var = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
tx = optax.adamw(learning_rate_fn, weight_decay=lambda_l2)
global_state = train_state.TrainState.create(
            apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
        )

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
options = orbax.checkpoint.CheckpointManagerOptions(create=True, save_interval_steps=val_every)
checkpoint_manager = orbax.checkpoint.CheckpointManager(save_model_folder_path, orbax_checkpointer, options)


if FINE_TUNE:
    restored = checkpoint_manager.restore(resume_model_folder_path)
    global_var['params'] = restored['model']['params']
    global_state = train_state.TrainState.create(
            apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
    )
    input_mean = jnp.array(restored['input_mean'])
    input_std = jnp.array(restored['input_std'])
else:
    input_mean = jnp.array(train_dataset.mean, dtype=jnp.float32)
    input_std = jnp.array(train_dataset.std, dtype=jnp.float32)

print("mean: ", input_mean.tolist())
print("std: ", input_std.tolist())


def apply_batch(var_collect, last_state, history, action, y, action_padding_mask, rngs):
    history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
    y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)
    
    global Run_first
    if Run_first == True:
        print("mean: ", input_mean.tolist())
        print("std: ", input_std.tolist())
        Run_first = False

    x = history[:, 1:, :]
    # tgt_mask = nn.Transformer.generate_square_subsequent_mask(action.size(1), device=action.device)
    y_pred = model.apply(var_collect, x, action, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=True) * input_std + input_mean
    last_pose = last_state[:, :6]
    for i in range(y_pred.shape[1]):
        # rotate dx, dy back to world frame
        y_pred_x = y_pred[:, i, 0] * jnp.cos(last_pose[:, 2]) - y_pred[:, i, 1] * jnp.sin(last_pose[:, 2])
        y_pred_y = y_pred[:, i, 0] * jnp.sin(last_pose[:, 2]) + y_pred[:, i, 1] * jnp.cos(last_pose[:, 2])
        y_pred = y_pred.at[:, i, 0].set(y_pred_x)
        y_pred = y_pred.at[:, i, 1].set(y_pred_y)
        # accumulate the poses
        y_pred = y_pred.at[:, i, :6].add(last_pose)
        y_pred = y_pred.at[:, i, 2].set(align_yaw_jax(y_pred[:, i, 2], 0.0))
        last_pose = y_pred[:, i, :6]
    return y_pred

@partial(jax.jit, static_argnums=(7,))
def loss_fn(state, var_collect, history, action, y, action_padding_mask, rngs, deterministic=False):
    history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
    y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)
    history = jax.lax.stop_gradient(history[:, 1:, :])
    y = jax.lax.stop_gradient(y)
    action = jax.lax.stop_gradient(action)

    y_pred = state.apply_fn(var_collect, history, action, history_padding_mask=None, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=deterministic)
    action_padding_mask_binary = (action_padding_mask == 0)[:, :, None]
    # loss_weight = (torch.arange(1, y_pred.shape[1] + 1, device=device, dtype=torch.float32) / y_pred.shape[1])[None, :, None]
    
    # add different state weights
    state_weights = jnp.array([0.5, 0.5, 2.0, 0.5, 0.0, 2.5])  # 每个状态的权重, X, Y, yaw, Vx, Vy, yawrate
    state_weights = state_weights[None, None, :]  # 调整形状为 (1, 1, 6)
    loss = jnp.mean(((y_pred - y) ** 2) * action_padding_mask_binary * state_weights)
    
    # add different weight for time step
    # time_loss_weight = (torch.arange(5, 1, -4 / y_pred.shape[1],dtype=torch.float32))[None, :] 
    # time_loss_weight = time_loss_weight / time_loss_weight.mean()
    
    # full_weights = torch.ones(y.shape[1], y.shape[2], dtype=torch.float32)[None, :, : ]
    # full_weights[:,:,0] = time_loss_weight
    # full_weights[:,:,1] = time_loss_weight
    # full_weights[:,:,2] = time_loss_weight
    # full_weights[:,:,3] = time_loss_weight
    # full_weights = jnp.array(full_weights.numpy())
    
    # loss = jnp.mean(((y_pred - y) ** 2) * action_padding_mask_binary * full_weights)

    # loss = jnp.mean(((y_pred - y) ** 2) * action_padding_mask_binary)
    # diff = (y_pred - y) * loss_weight
    # loss = torch.mean(torch.masked_select(diff, action_padding_mask_binary) ** 2)
    
    return loss


def val_episode(var_collect, episode_num, rngs, dateset):
    episode = dateset.get_episode(episode_num)
    episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask = dateset[episode_num:episode_num+1]
    history = jnp.array(history.numpy())
    action = jnp.array(action.numpy())
    y = jnp.array(y.numpy())
    action_padding_mask = jnp.array(action_padding_mask.numpy())
    predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs)
    return np.array(predicted_states)
    

def val_loop(state, var_collect, val_loader, rngs):
    val_loss = 0.0
    t_val = tqdm.tqdm(val_loader)
    for i, (history, action, y, action_padding_mask) in enumerate(t_val):
        history = jnp.array(history.numpy())
        action = jnp.array(action.numpy())
        y = jnp.array(y.numpy())
        action_padding_mask = jnp.array(action_padding_mask.numpy())
        val_loss += loss_fn(state, var_collect, history, action, y, action_padding_mask, global_rngs, True).item()
        t_val.set_description(f'Validation Loss: {(val_loss / (i + 1)):.4f}')
        t_val.refresh()
    val_loss /= len(val_loader)
    return val_loss

def visualize_episode(epoch_num: int, episode_num, val_dataset, rngs):
    val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(os.path.join(save_model_folder_path, f"{epoch_num}", "default"))
    # import ipdb; ipdb.set_trace()
    val_collect['params'] = raw_restored['model']['params']
    predicted_states = val_episode(val_collect, episode_num, rngs, val_dataset)
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

def clear_directory(path):
    files = glob.glob(os.path.join(path, '*'))
    
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"无法删除 {f}: {e}")

def compute_1d_rmse(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape
    mse = torch.mean((tensor1 - tensor2) ** 2)
    rmse = torch.sqrt(mse).item()
    return rmse

def compute_vertor_rmse(tensor1_x, tensor1_y, tensor2_x, tensor2_y):
    assert tensor1_x.shape == tensor2_x.shape
    assert tensor1_y.shape == tensor2_y.shape
    x_error = tensor1_x - tensor2_x
    y_error = tensor1_y - tensor2_y
    position_error = x_error ** 2 + y_error **2
    mse = torch.mean(position_error)
    rmse = torch.sqrt(mse).item()
    return rmse

def temp_use_verify_data():
    dataset_path = check_data_path  #10 pkl
    fig_result_path = '/home/gzh/anycar/model_test_result_fig'
    dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
    test_dataset = MujocoDataset(dataset_files, history_length, prediction_length, delays=None, teacher_forcing=False, binary_mask=False,attack=False, use_zero_point=USE_ZERO_POINT)
    data_num = len(test_dataset)

    global input_mean
    global input_std
    global Run_first
    Run_first = True

    rmse_dict = {
        "position_error":[],
        "v_error":[],
        "yawrate_error":[]
    }

    clear_directory(fig_result_path + "/")

    val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(os.path.join(save_model_folder_path, f"{num_epochs}", "default"))
    # import ipdb; ipdb.set_trace()
    val_collect['params'] = raw_restored['model']['params']

    def model_hash(params):
        md5 = hashlib.md5()
        flat_params, _ = tree_flatten(params)
        for param in flat_params:
            md5.update(jnp.array(param).tobytes())
        return md5.hexdigest()

    print(model_hash(val_collect['params']))

    for epoch in range(data_num):

        if epoch + 1 >= len(test_dataset):
            continue

        predicted_states = val_episode(val_collect, epoch + 1, global_rngs, test_dataset)
        episode = test_dataset.get_episode(epoch + 1)

        # calculate value rmse
        position_error_rmse = compute_vertor_rmse(episode[:, 0][-prediction_length:], episode[:, 1][-prediction_length:], predicted_states[0, :, 0], predicted_states[0, :, 1])
        v_error_rmse = compute_vertor_rmse(episode[:, 3][-prediction_length:],episode[:, 4][-prediction_length:], predicted_states[0, :, 3], predicted_states[0, :, 4])
        yaw_error_rmse = compute_1d_rmse(episode[:, 5][-prediction_length:], predicted_states[0, :, 5])

        rmse_dict["position_error"].append(position_error_rmse)
        rmse_dict["v_error"].append(v_error_rmse)
        rmse_dict["yawrate_error"].append(yaw_error_rmse)

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

        axs[1, 0].plot(episode[:, 2] * 57.3, label='Ground Truth yaw(deg)')
        axs[1, 0].plot(predict_x, predicted_states[0, :, 2] * 57.3, label='Predicted yaw(deg)')
        axs[1, 0].legend()

        axs[1, 1].plot(episode[:, 5] * 57.3, label='Ground Truth yawrate(deg)')
        axs[1, 1].plot(predict_x, predicted_states[0, :, 5] * 57.3, label='Predicted yawrate(deg)')
        axs[1, 1].legend()

        fig.tight_layout()
        plt.title("prediction result data_" + str(epoch+1))
        save_path = os.path.join(fig_result_path, "result" + str(epoch+1))
        fig.savefig(save_path, format="png")
        plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

    axs[0].scatter(range(len(rmse_dict["position_error"])), rmse_dict["position_error"], marker='o')
    axs[0].set_title("position_error rmse")

    axs[1].scatter(range(len(rmse_dict["position_error"])), rmse_dict["v_error"], marker='o')
    axs[1].set_title("v_error rmse")

    axs[2].scatter(range(len(rmse_dict["position_error"])), rmse_dict["yawrate_error"], marker='o')
    axs[2].set_title("yawrate_error rmse")

    mean_rmse = [np.mean(rmse_dict["position_error"]),np.mean(rmse_dict["v_error"]),np.mean(rmse_dict["yawrate_error"])]
    print("position_error = " + str(np.mean(rmse_dict["position_error"])))
    print("v_error = " + str(np.mean(rmse_dict["v_error"])))
    print("yawrate_error = " + str(np.mean(rmse_dict["yawrate_error"])))

    axs[3].barh(list(rmse_dict.keys()), mean_rmse, color='skyblue')
    axs[3].set_title("mean RMSE fron all verify data")
    axs[3].set_xlabel("Values")

    fig.tight_layout()
    save_path = os.path.join(fig_result_path, "final result")
    fig.savefig(save_path, format="png")
    plt.show()

train_losses = []
val_losses = []
val_epoch_nums = []

for epoch in range(num_epochs):
    running_loss = 0.0
    t = tqdm.tqdm(train_loader)

    for i, (history, action, y, action_padding_mask) in enumerate(t):
        history = jnp.array(history.numpy())
        action = jnp.array(action.numpy())
        y = jnp.array(y.numpy())
        action_padding_mask = jnp.array(action_padding_mask.numpy())

        def this_loss_fn(var_collect, history, action, y, action_padding_mask):
            return loss_fn(global_state, var_collect, history, action, y, action_padding_mask, global_rngs)

        grad_fn = jax.value_and_grad(this_loss_fn, has_aux=False)

        loss, grads = grad_fn(global_var, history, action, y, action_padding_mask)
        loss_item = loss.item()
        running_loss += loss_item

        t.set_description(f'Epoch {epoch + 1}, Loss: {(running_loss / (i + 1)):.4f}, LR: {learning_rate_fn(global_state.step):.6f}')
        t.refresh()

        global_state = global_state.apply_gradients(grads=grads["params"])
        global_var['params'] = global_state.params

    running_loss /= len(train_loader)
    train_losses.append(running_loss)
    wandb.log({"train_loss": running_loss, "learning_rate": learning_rate_fn(global_state.step)})
    print(save_model_folder_path)
    # import ipdb; ipdb.set_trace()
    ckpt = {'model': global_state, 'input_mean': input_mean, 'input_std': input_std}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(epoch+1, ckpt, save_kwargs={'save_args': save_args})

    if (epoch + 1) % val_every == 0:
        visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
        val_loss = val_loop(global_state, global_var, val_loader, global_rngs)
        val_losses.append(val_loss)
        val_epoch_nums.append(epoch + 1)
        print(f'Validation Loss: {val_loss:.4f}')
        wandb.log({"val_loss": val_loss})

    if (epoch+1) % num_epochs == 0:
        temp_use_verify_data()

train_epoch_nums = list(range(1, num_epochs + 1))
plt.figure()
plt.plot(train_epoch_nums, train_losses, label='Train Loss')
plt.plot(val_epoch_nums, val_losses, label='Val Loss')
plt.legend()
plt.savefig('train_val_loss.png')
# plt.show()

# model.eval()
visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
test_loss = val_loop(global_state, global_var, test_loader, global_rngs)
print(f'Test Loss: {test_loss:.4f}')

# Save the model
# torch.save(model.state_dict(), 'model.pth')
wandb.save('model_checkpoint/')

wandb.finish()
