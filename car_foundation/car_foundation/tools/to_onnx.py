import torch
import torch.onnx
import torch.nn as nn

from car_foundation.models import TorchTransformerDecoder, TorchGPT2

# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-01-24T14:18:55.571-model_checkpoint"
# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-02-10T10:20:21.689-model_checkpoint"
# model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-03-20T16:56:16.174-model_checkpoint"
model_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/2025-03-20T19:50:30.419-model_checkpoint"
output_path = "/disk1/collect_data_from_anycar/Compare_pytorch_and_jax/"

# state_dim = 6
# action_dim = 2
# latent_dim = 64
# num_heads = 4
# num_layers = 2
# dropout = 0.1

state_dim = 6
action_dim = 2
latent_dim = 256 #128 #64
num_heads = 4
num_layers = 3 #2
dropout = 0.1

history_length = 250
prediction_length = 50
batch_size = 1

# Device to use
device = torch.device("cuda")
assert device.type == "cuda", "Only cuda is supported"

checkpoint = torch.load(model_path)

model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout, history_length, prediction_length).to(device)
model.eval()
model.load_state_dict(checkpoint['model_state_dict'])

history_input = torch.randn(batch_size, history_length, state_dim + action_dim).to(device)
history_mask = torch.ones(batch_size, history_length * 2 - 1).to(device)
prediction_input = torch.randn(batch_size, prediction_length, action_dim).to(device)
prediction_mask = torch.ones(batch_size, prediction_length).to(device)

# 导出模型到 ONNX 格式
output_onnx_path = output_path + "torch_transformer_decoder_fix_batch_size_1_0321.onnx"
torch.onnx.export(
    model, 
    (history_input, prediction_input, history_mask, prediction_mask), 
    output_onnx_path, 
    export_params=True, 
    opset_version=14, 
    do_constant_folding=True, 
    input_names=['history_input', 'prediction_input', 'history_mask', 'prediction_mask'], 
    output_names=['output'], 
    dynamic_axes={}
)

print(f"模型已成功导出为 {output_onnx_path}")
