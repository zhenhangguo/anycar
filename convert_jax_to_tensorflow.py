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

import tensorflow as tf
from jax.experimental import jax2tf


Save_Fig = False

model_path = "/disk1/collect_data_from_anycar/temp_verify_backlash_model/2025-01-14T18:44:21.817-model_checkpoint"
dataset_path = '/disk1/collect_data_from_anycar/New_demo/check_data'
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

# print("input_mean = " + str(input_mean))
# print("input_std = " + str(input_std))
# print("model_hash = " + str(model_hash(val_collect['params'])))

def model_fn(x):
    state = x[:, :history_length -1 , :state_dim + action_dim]
    action = x[:, history_length-1:history_length+prediction_length-1, 6:8]
    return model.apply(val_collect, state, action, rngs=global_rngs, deterministic=True)

tf_model = jax2tf.convert(model_fn)

input_signature = tf.TensorSpec(shape=(batch_size, history_length - 1 + prediction_length, state_dim + action_dim), dtype=tf.float32)
@tf.function(input_signature=[input_signature])
def tf_infer(x):
    return tf_model(x)

tf.saved_model.save(tf_infer, '/disk1/collect_data_from_anycar/New_demo')

