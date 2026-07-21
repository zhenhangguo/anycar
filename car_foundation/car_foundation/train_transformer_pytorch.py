from functools import partial
import json
import os
import subprocess
import sys
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import numpy as np
import optax
import orbax

SOURCE_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCE_PACKAGE_ROOT = os.path.join(SOURCE_REPO_ROOT, 'car_foundation')
for path in (SOURCE_REPO_ROOT, SOURCE_PACKAGE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_foundation.dataset import DynamicsDataset, IssacSimDataset, MujocoDataset
from car_foundation.models import (
    TorchGPT2,
    TorchMLP,
    TorchTransformer,
    TorchTransformerDecoder,
    TorchTransformerDecoderCurrentState,
    TorchTransformerDecoderCurrentStateMLP,
)
from car_foundation.utils import generate_subsequences, generate_subsequences_hf, align_yaw, align_yaw_jax
import datetime
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

from rich.progress import track

PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"

torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")

def env_int(name, default):
    return int(os.environ.get(name, default))

def env_float(name, default):
    return float(os.environ.get(name, default))

def env_str(name, default):
    return os.environ.get(name, default)

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

FINE_TUNE = env_bool("ANYCAR_FINE_TUNE", True)

if FINE_TUNE:
    lr_begin = env_float("ANYCAR_LR_BEGIN", 5e-4)
    warmup_period = env_int("ANYCAR_WARMUP_PERIOD", 2)
    num_epochs = env_int("ANYCAR_NUM_EPOCHS", 400)
    load_checkpoint = True
    resume_model_path = env_str("ANYCAR_RESUME_MODEL_PATH", "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/test_compress_file/torch_model_400_use_cnn_before_emb")
else:
    lr_begin = env_float("ANYCAR_LR_BEGIN", 5e-4)
    warmup_period = env_int("ANYCAR_WARMUP_PERIOD", 500)
    num_epochs = env_int("ANYCAR_NUM_EPOCHS", 400)
    load_checkpoint = False

val_every = env_int("ANYCAR_VAL_EVERY", 20)
batch_size = env_int("ANYCAR_BATCH_SIZE", 512)
lambda_l2 = env_float("ANYCAR_LAMBDA_L2", 1e-4)
dataset_path = env_str("ANYCAR_DATASET_PATH", '/disk1/collect_data_from_anycar/data_from_bag/new_temp_data/c2_bag_04')

# dataset_path = '/disk1/collect_data_from_anycar/New_demo/new_data_with_x_mean_zero/total_data_1'
# dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/temp_debug_data'
# dataset_path = '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-01-14T16:39:46.443-nuplan-dynamic-model-base'
# check_data_path = '/disk1/collect_data_from_anycar/temp_verify_backlash_model/2025-01-14T18:15:29.673-nuplan-dynamic-model-verify'
check_data_path = env_str("ANYCAR_CHECK_DATA_PATH", '/disk1/collect_data_from_anycar/New_demo/check_data_with_offset')
comment = env_str("ANYCAR_COMMENT", 'torch')

# Device to use
device = torch.device("cuda")
assert device.type == "cuda", "Only cuda is supported"

num_workers = env_int("ANYCAR_NUM_WORKERS", 6)

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1
model_variant = env_str("ANYCAR_MODEL_VARIANT", "baseline")
current_dim = env_int("ANYCAR_CURRENT_DIM", state_dim + action_dim)
fusion_hidden_dim = env_int("ANYCAR_FUSION_HIDDEN_DIM", 128)
if model_variant not in ("baseline", "current_concat_linear", "current_concat_mlp"):
    raise ValueError(f"Unsupported ANYCAR_MODEL_VARIANT={model_variant}")

def is_project_root(path):
    return path and os.path.isfile(os.path.join(path, 'set_env.sh')) and os.path.isdir(os.path.join(path, 'car_foundation'))

def resolve_project_root():
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    candidates = [
        os.environ.get('CAR_PATH'),
        source_root,
        os.getcwd(),
    ]
    for path in candidates:
        if is_project_root(path):
            return os.path.abspath(path)
    return source_root

save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
PROJECT_ROOT = resolve_project_root()
OUTPUT_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints')
OUTPUT_FIGURE_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
save_model_folder_path = os.path.join(OUTPUT_CHECKPOINT_DIR, f'{save_model_folder_prefix}-model_checkpoint')
os.makedirs(OUTPUT_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURE_DIR, exist_ok=True)
os.makedirs(save_model_folder_path, exist_ok=True)

RUN_START_TIME = datetime.datetime.now().isoformat(timespec='seconds')
RUN_START_WALL_TIME = time.time()
RUN_SUMMARY_JSON = os.path.join(save_model_folder_path, 'run_summary.json')
RUN_SUMMARY_TXT = os.path.join(save_model_folder_path, 'run_summary.txt')
SPLIT_FILE_PATHS = {
    'train': os.path.join(save_model_folder_path, 'train_files.txt'),
    'val': os.path.join(save_model_folder_path, 'val_files.txt'),
    'test': os.path.join(save_model_folder_path, 'test_files.txt'),
}

def now_iso():
    return datetime.datetime.now().isoformat(timespec='seconds')

def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ''

def tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value

def write_list_file(path, values):
    with open(path, 'w') as f:
        for value in values:
            f.write(f'{value}\n')

def get_wandb_info():
    run = wandb.run
    if run is None:
        return {}
    url = ''
    try:
        url = run.get_url()
    except Exception:
        url = getattr(run, 'url', '')
    return {
        'project': getattr(run, 'project', ''),
        'entity': getattr(run, 'entity', ''),
        'id': getattr(run, 'id', ''),
        'name': getattr(run, 'name', ''),
        'url': url,
    }

def write_run_summary(
    status,
    train_losses,
    val_losses,
    val_epoch_nums,
    checkpoint_records,
    input_mean=None,
    input_std=None,
    latest_checkpoint='',
    best_val_record=None,
    latest_epoch=0,
    latest_train_loss=None,
    latest_val_loss=None,
    end_time='',
):
    summary = {
        'status': status,
        'start_time': RUN_START_TIME,
        'updated_time': now_iso(),
        'end_time': end_time,
        'elapsed_seconds': time.time() - RUN_START_WALL_TIME,
        'git_commit': get_git_commit(),
        'script_path': os.path.abspath(__file__),
        'project_root': PROJECT_ROOT,
        'output_dir': save_model_folder_path,
        'summary_json': RUN_SUMMARY_JSON,
        'summary_txt': RUN_SUMMARY_TXT,
        'figure_dir': OUTPUT_FIGURE_DIR,
        'latest_checkpoint': latest_checkpoint,
        'best_val_checkpoint': '' if best_val_record is None else best_val_record.get('checkpoint_path', ''),
        'best_val_loss': None if best_val_record is None else best_val_record.get('val_loss'),
        'latest_epoch': latest_epoch,
        'latest_train_loss': latest_train_loss,
        'latest_val_loss': latest_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'config': {
            'architecture': architecture,
            'model_variant': model_variant,
            'current_dim': current_dim,
            'fusion_hidden_dim': fusion_hidden_dim if model_variant == "current_concat_mlp" else None,
            'fine_tune': FINE_TUNE,
            'load_checkpoint': load_checkpoint,
            'resume_model_path': resume_model_path if load_checkpoint else '',
            'dataset_path': dataset_path,
            'check_data_path': check_data_path,
            'history_length': history_length,
            'prediction_length': prediction_length,
            'actual_input_length': actual_input_length,
            'compress_sample': Compress_Sample,
            'compress_history_length': compress_history_length,
            'lr_begin': lr_begin,
            'warmup_period': warmup_period,
            'num_epochs': num_epochs,
            'val_every': val_every,
            'batch_size': batch_size,
            'lambda_l2': lambda_l2,
            'num_workers': num_workers,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'latent_dim': latent_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'attack': ATTACK,
            'use_zero_point': USE_ZERO_POINT,
            'teacher_forcing': teacher_forcing,
            'binary_mask': binary_mask,
            'comment': comment,
        },
        'dataset': {
            'total_pkl_files': total_len,
            'train_pkl_files': len(data_70),
            'val_pkl_files': len(data_20),
            'test_pkl_files': len(data_10),
            'train_dataset_len': len(train_dataset),
            'val_dataset_len': len(val_dataset),
            'test_dataset_len': len(test_dataset),
            'split_files': SPLIT_FILE_PATHS,
        },
        'normalization': {
            'input_mean': tensor_to_list(input_mean),
            'input_std': tensor_to_list(input_std),
        },
        'wandb': get_wandb_info(),
        'environment': {
            key: os.environ.get(key)
            for key in sorted(os.environ)
            if key.startswith('ANYCAR_') or key in ('JAX_PLATFORM_NAME', 'CUDA_VISIBLE_DEVICES')
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_epoch_nums': val_epoch_nums,
        'checkpoints': checkpoint_records,
    }
    with open(RUN_SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)

    lines = [
        f"status: {summary['status']}",
        f"start_time: {summary['start_time']}",
        f"updated_time: {summary['updated_time']}",
        f"end_time: {summary['end_time']}",
        f"elapsed_seconds: {summary['elapsed_seconds']}",
        f"model_variant: {model_variant}",
        f"dataset_path: {dataset_path}",
        f"resume_model_path: {resume_model_path if load_checkpoint else ''}",
        f"output_dir: {save_model_folder_path}",
        f"latest_checkpoint: {summary['latest_checkpoint']}",
        f"best_val_checkpoint: {summary['best_val_checkpoint']}",
        f"latest_epoch: {summary['latest_epoch']}",
        f"latest_train_loss: {summary['latest_train_loss']}",
        f"latest_val_loss: {summary['latest_val_loss']}",
        f"best_val_loss: {summary['best_val_loss']}",
        f"wandb_url: {summary['wandb'].get('url', '')}",
        f"summary_json: {RUN_SUMMARY_JSON}",
    ]
    with open(RUN_SUMMARY_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')

architecture = 'torch_decoder'
use_torch_decoder = True

if Compress_Sample:
    actual_input_length = int(actual_input_length * compress_ratio)

def create_decoder_model():
    model_cls = TorchTransformerDecoder
    extra_kwargs = {}
    if model_variant == "current_concat_linear":
        model_cls = TorchTransformerDecoderCurrentState
        extra_kwargs["current_dim"] = current_dim
    elif model_variant == "current_concat_mlp":
        model_cls = TorchTransformerDecoderCurrentStateMLP
        extra_kwargs["current_dim"] = current_dim
        extra_kwargs["fusion_hidden_dim"] = fusion_hidden_dim

    if Compress_Sample:
        return model_cls(
            state_dim,
            action_dim,
            state_dim,
            latent_dim,
            num_heads,
            num_layers,
            device,
            dropout,
            compress_history_length,
            prediction_length,
            actual_input_length,
            **extra_kwargs,
        ).to(device)
    return model_cls(
        state_dim,
        action_dim,
        state_dim,
        latent_dim,
        num_heads,
        num_layers,
        device,
        dropout,
        **extra_kwargs,
    ).to(device)

def load_training_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    if model_variant == "baseline":
        model.load_state_dict(state_dict)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"loaded checkpoint with missing={missing}, unexpected={unexpected}")
        allowed_missing = {
            "current_concat_linear": {
                "action_fusion.weight",
                "action_fusion.bias",
            },
            "current_concat_mlp": {
                "action_fusion.0.weight",
                "action_fusion.0.bias",
                "action_fusion.2.weight",
                "action_fusion.2.bias",
            },
        }[model_variant]
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        if missing and not set(missing).issubset(allowed_missing):
            raise RuntimeError(f"Unexpected missing checkpoint keys: {missing}")
        if set(missing) == allowed_missing:
            model.init_fusion_from_action_embedding()
            print(f"initialized {model_variant} fusion to preserve baseline action embedding")
    return checkpoint

if architecture == "torch_decoder":
    model = create_decoder_model()

# Load the dataset
binary_mask = False
dataset_files = sorted(glob.glob(os.path.join(dataset_path, '*.pkl'))) # get all *.pkl file in this path
random.shuffle(dataset_files)
total_len = len(dataset_files)
split_70 = int(total_len * 0.7)
split_80 = int(total_len * 0.8)
split_20 = int(total_len * 0.9)
split_5 = int(total_len * 0.95)
data_70 = dataset_files[:split_80]
data_20 = dataset_files[split_80:split_5]
data_10 = dataset_files[split_5:]
write_list_file(SPLIT_FILE_PATHS['train'], data_70)
write_list_file(SPLIT_FILE_PATHS['val'], data_20)
write_list_file(SPLIT_FILE_PATHS['test'], data_10)

train_dataset = MujocoDataset(data_70, history_length+1, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT)
print("train data length", len(train_dataset))

val_dataset = MujocoDataset(data_20, history_length+1, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK, use_zero_point=USE_ZERO_POINT)
test_dataset = MujocoDataset(data_10, history_length+1, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK, use_zero_point=USE_ZERO_POINT)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True)

num_steps_per_epoch = len(train_loader)

wandb.init(
    # set the wandb project where this run will be logged
    project="transformer-sequence-prediction",
    name=f"{architecture}-{model_variant}",

    # track hyperparameters and run metadata
    config={
        "history_length": history_length,
        "prediction_length": prediction_length,
        "delays": delays,
        "teacher_forcing": teacher_forcing,

        "learning_rate": lr_begin,
        "warmup_period": warmup_period,
        "architecture": architecture,
        "model_variant": model_variant,
        "current_dim": current_dim,
        "fusion_hidden_dim": fusion_hidden_dim if model_variant == "current_concat_mlp" else None,
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
        "resume_model_path": resume_model_path if load_checkpoint else "",
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
            last_pose = y_pred[:, i, :6]

        return y_pred
def val_episode(model, episode_num, dateset):
    episode = dateset.get_episode(episode_num)
    episode = torch.unsqueeze(episode, 0)
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask = dateset[episode_num:episode_num+1]
    # MujocoDataset receives history_length + 1 frames.  After dropping the
    # invalid first delta, the final observed absolute state is at this index.
    predicted_states = apply_batch_torch(model, batch[:, history_length, :], history, action, y, input_mean, input_std)
    return np.array(predicted_states.cpu())

def val_loop(model_path, val_loader):
    model = create_decoder_model()
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
    model = create_decoder_model()
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
    episode_fig_path = os.path.join(OUTPUT_FIGURE_DIR, 'episode.png')
    fig.savefig(episode_fig_path)
    plt.close(fig)
    wandb.log({"episode": wandb.Image(episode_fig_path)})

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
    resume_checkpoint = load_training_checkpoint(model, resume_model_path)
    input_mean = resume_checkpoint['input_mean']
    input_std = resume_checkpoint['input_std']
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
checkpoint_records = []
best_val_record = None
latest_checkpoint_path = ''
latest_val_loss = None
print(f"run output dir: {save_model_folder_path}")
print(f"run summary: {RUN_SUMMARY_JSON}")
write_run_summary(
    status='running',
    train_losses=train_losses,
    val_losses=val_losses,
    val_epoch_nums=val_epoch_nums,
    checkpoint_records=checkpoint_records,
    input_mean=input_mean,
    input_std=input_std,
)
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

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    wandb.log({"train_loss": train_loss, "learning_rate": learning_rate_fn(global_step)})
    print("train_loss = " + str(train_loss))

    if (epoch + 1) % val_every == 0:

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'architecture': architecture,
            'model_variant': model_variant,
            'current_dim': current_dim,
            'fusion_hidden_dim': fusion_hidden_dim if model_variant == "current_concat_mlp" else None,
            'dataset_path': dataset_path,
            'resume_model_path': resume_model_path if load_checkpoint else '',
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
        latest_checkpoint_path = save_model_folder_path_epoch
        checkpoint_record = {
            'epoch': epoch + 1,
            'checkpoint_path': save_model_folder_path_epoch,
            'train_loss': train_loss,
            'val_loss': None,
            'learning_rate': float(learning_rate_fn(global_step)),
            'saved_time': now_iso(),
        }
        checkpoint_records.append(checkpoint_record)
        write_run_summary(
            status='running',
            train_losses=train_losses,
            val_losses=val_losses,
            val_epoch_nums=val_epoch_nums,
            checkpoint_records=checkpoint_records,
            input_mean=input_mean,
            input_std=input_std,
            latest_checkpoint=latest_checkpoint_path,
            best_val_record=best_val_record,
            latest_epoch=epoch + 1,
            latest_train_loss=train_loss,
            latest_val_loss=latest_val_loss,
        )

        visualize_episode(1, val_dataset, save_model_folder_path_epoch)
        val_loss = val_loop(save_model_folder_path_epoch, val_loader)
        latest_val_loss = val_loss
        val_losses.append(val_loss)
        val_epoch_nums.append(epoch + 1)
        checkpoint_record['val_loss'] = val_loss
        checkpoint_record['validated_time'] = now_iso()
        if best_val_record is None or val_loss < best_val_record['val_loss']:
            best_val_record = checkpoint_record.copy()
        print(f'Validation Loss: {val_loss:.4f}')
        wandb.log({"val_loss": val_loss})
        write_run_summary(
            status='running',
            train_losses=train_losses,
            val_losses=val_losses,
            val_epoch_nums=val_epoch_nums,
            checkpoint_records=checkpoint_records,
            input_mean=input_mean,
            input_std=input_std,
            latest_checkpoint=latest_checkpoint_path,
            best_val_record=best_val_record,
            latest_epoch=epoch + 1,
            latest_train_loss=train_loss,
            latest_val_loss=latest_val_loss,
        )
    else:
        write_run_summary(
            status='running',
            train_losses=train_losses,
            val_losses=val_losses,
            val_epoch_nums=val_epoch_nums,
            checkpoint_records=checkpoint_records,
            input_mean=input_mean,
            input_std=input_std,
            latest_checkpoint=latest_checkpoint_path,
            best_val_record=best_val_record,
            latest_epoch=epoch + 1,
            latest_train_loss=train_loss,
            latest_val_loss=latest_val_loss,
        )

train_epoch_nums = list(range(1, len(train_losses) + 1))
plt.figure()
plt.plot(train_epoch_nums, train_losses, label='Train Loss')
plt.plot(val_epoch_nums, val_losses, label='Val Loss')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FIGURE_DIR, 'train_val_loss.png'))
# plt.show()

write_run_summary(
    status='completed',
    train_losses=train_losses,
    val_losses=val_losses,
    val_epoch_nums=val_epoch_nums,
    checkpoint_records=checkpoint_records,
    input_mean=input_mean,
    input_std=input_std,
    latest_checkpoint=latest_checkpoint_path,
    best_val_record=best_val_record,
    latest_epoch=num_epochs,
    latest_train_loss=train_losses[-1] if train_losses else None,
    latest_val_loss=latest_val_loss,
    end_time=now_iso(),
)
print(f"run summary: {RUN_SUMMARY_JSON}")
wandb.finish()
