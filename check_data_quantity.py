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
from generate_data_utils import *


model_path = "/home/gzh/Desktop/anycar/anycar/car_foundation/car_foundation/models/2024-11-19-model_checkpoint/"

# dataset_path = '/home/gzh/Desktop/anycar/anycar/car_foundation/car_foundation/data/data_verify'  #10 pkl
dataset_path = '/disk1/collect_data_from_anycar/2025-03-11T17:19:31.119-nuplan-dynamic-model'  #10 pkl
Long_Path_sim = False
fig_result_path = '/home/gzh/Desktop/anycar/anycar/model_test_result_fig'

model_checkpint = 400

history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False
binary_mask = False
ATTACK = False  # verify data will not add noise
USE_ZERO_POINT = True  # set zero points as trajectory first point

# debug choose
Show_Picture = True

# funciton_1 : clear data in pkl !!!!!
# clear_unfit_pkl_file("/disk1/collect_data_from_anycar/2025-01-14T16:39:46.443-nuplan-dynamic-model")

# funciton_2 : plot data density
# data_analysis_mean, data_analysis_std = get_dict_mean_and_std()
dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl')) # get all *.pkl file in this path
test_dataset = MujocoDataset(dataset_files, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, use_zero_point=USE_ZERO_POINT)
# calculate_data_range(test_dataset, data_analysis_mean, data_analysis_std)
# if Show_Picture:
#     show_picture(data_analysis_mean, data_analysis_std)


plot_data_density(test_dataset)

