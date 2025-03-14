import numpy as np
import onnxruntime as ort
import glob
import os
import onnx


from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR
from car_foundation.dataset import MujocoDataset
from verify_utils import *

Save_Fig = True

onnx_model_path = "/home/gzh/Desktop/torch_transformer_decoder_fix_batch_size_512_0311_sanitize.onnx"
# dataset_path =  '/disk1/collect_data_from_anycar/New_demo/check_data_with_offset' 
dataset_path =  '/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/temp_debug_data'  #10 pkl
# dataset_path = '/disk1/collect_data_from_anycar/check_data/verify_bag_data_0310'


# *************************Load checkpoint for mean and std******************
# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-02-10T10:20:21.689-model_checkpoint"
# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-01-24T14:18:55.571-model_checkpoint"
model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-03-11T19:25:14.129-model_checkpoint"


fig_result_path = '/home/gzh/anycar/model_test_result_fig'

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
USE_ZERO_POINT = True

# Load ONNX model
ort_session = ort.InferenceSession(onnx_model_path)

checkpoint = torch.load(model_path, weights_only=True)

# set mean and std for checkpoint or other data
input_mean = checkpoint['input_mean'].cpu().numpy()
input_std = checkpoint['input_std'].cpu().numpy()


model = onnx.load(onnx_model_path)
# 计算参数量
total_params = 0
for tensor in model.graph.initializer:
    param_count = 1
    for dim in tensor.dims:
        param_count *= dim
    total_params += param_count

print(f"总参数量: {total_params} (≈ {total_params / 1e6:.2f}M)")


