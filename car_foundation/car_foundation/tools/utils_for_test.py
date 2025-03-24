import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os
import glob
import tqdm
import concurrent.futures

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

    return roll, pitch, yaw

def generate_subsequences(input_tensor):
    """
    Generates subsequences from the input tensor with increasing length,
    pads them to full length, and generates a padding mask.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (N, S, E) where
        N is the batch size,
        S is the sequence length,
        E is the vector dimension.

    Returns:
    tuple: A tuple containing:
        - output_tensors (torch.Tensor): Tensor of shape (N * S, S, E)
        where each sub-tensor includes subsequences padded to full length.
        - mask (torch.Tensor): Float32 mask of shape (N * S, S) indicating
        paddings (-inf for padding, 0 for data).
    """
    N, S, E = input_tensor.shape
    # Initialize a tensor to hold the padded subsequences
    output_tensors = torch.zeros(N, S, S, E, device=input_tensor.device, dtype=input_tensor.dtype)
    mask = torch.fill_(torch.zeros(N, S, S, dtype=torch.float32, device=input_tensor.device), float('-inf'))

    # Loop over each possible subsequence length
    for i in range(S):
        output_tensors[:, i, :i+1, :] = input_tensor[:, :i+1, :]
        mask[:, i, :i+1] = 0.0

    return output_tensors.view(N * S, S, E), mask.view(N * S, S)

def generate_subsequences_hf(input_tensor):
    """
    Generates subsequences from the input tensor with increasing length,
    pads them to full length, and generates a padding mask.
    HuggingFace convention: mask is 1 for data and 0 for padding.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (N, S, E) where
        N is the batch size,
        S is the sequence length,
        E is the vector dimension.

    Returns:
    tuple: A tuple containing:
        - output_tensors (torch.Tensor): Tensor of shape (N * S, S, E)
        where each sub-tensor includes subsequences padded to full length.
        - mask (torch.Tensor): Float32 mask of shape (N * S, S) indicating
        paddings (0 for padding, 1 for data).
    """
    N, S, E = input_tensor.shape
    # Initialize a tensor to hold the padded subsequences
    output_tensors = torch.zeros(N, S, S, E, device=input_tensor.device, dtype=input_tensor.dtype)
    mask = torch.zeros(N, S, S, dtype=torch.float32, device=input_tensor.device)

    # Loop over each possible subsequence length
    for i in range(S):
        output_tensors[:, i, :i+1, :] = input_tensor[:, :i+1, :]
        mask[:, i, :i+1] = 1.0

    return output_tensors.view(N * S, S, E), mask.view(N * S, S)

def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))
    return d_yaw_aligned + yaw_2

class MujocoDataset(Dataset):
    def __init__(self, path, history_length, action_length, 
                 delays=None, 
                 mean=None, 
                 std=None, 
                 teacher_forcing=True, 
                 binary_mask=False, 
                 use_jax=False,
                 attack=False,
                 filter=False,
                 add_noise=False,
                 use_zero_point=False,
    ):
        self.attack = attack
        self.add_noise = add_noise
        if delays is not None and any([d < 0 for d in delays]):
            raise ValueError('Delay should be greater than or equal to 0')

        
        def load_pickle(file):
            try:
                mujoco_raw_dataset = pickle.load(open(file, 'rb'))
            except:
                raise ValueError(f'Error loading the pickle file: {file}')
            q = np.array([mujoco_raw_dataset.data_logs["xori_w"],
                        mujoco_raw_dataset.data_logs["xori_x"],
                        mujoco_raw_dataset.data_logs["xori_y"],
                        mujoco_raw_dataset.data_logs["xori_z"]]).T
            _, _, y = quaternion_to_euler(q)
            data_array = np.array(
                [mujoco_raw_dataset.data_logs["xpos_x"],
                mujoco_raw_dataset.data_logs["xpos_y"],
                y,
                mujoco_raw_dataset.data_logs["xvel_x"],
                mujoco_raw_dataset.data_logs["xvel_y"],
                mujoco_raw_dataset.data_logs["avel_z"],
                mujoco_raw_dataset.data_logs["throttle"],
                mujoco_raw_dataset.data_logs["steer"],
                np.zeros_like(mujoco_raw_dataset.data_logs["xpos_x"]),
                ]).T
            episode_length = np.where(mujoco_raw_dataset.data_logs["lap_end"] == 1)[0][0] + 1

            # print("mujoco_raw_dataset.data_logs[xvel_y] = " + str(mujoco_raw_dataset.data_logs["xvel_y"]))

            # print("EPISODE LENGTH", episode_length)            
            episode_terminations = np.arange(episode_length - 1, data_array.shape[0], episode_length)
            assert np.all(mujoco_raw_dataset.data_logs["lap_end"][episode_terminations] == 1), 'Episode terminations are not correct'
            data_array = data_array.reshape(-1, episode_length, data_array.shape[1])
            # shift the data by the delay
            if delays:
                data_array_delayed = []
                max_delay = max(delays)
                for delay in delays:
                    if delay == 0:
                        data_array_delayed.append(data_array[:, max_delay:, :])
                    else:
                        data_array_delayed.append(
                            np.concatenate([
                                data_array[:, :-delay, :6],
                                data_array[:, delay:, 6:],
                            ], axis=2)[:, max_delay-delay:, :]
                        )
                data_array = np.concatenate(data_array_delayed, axis=0)

            # remove the last few steps to make the data divisible by the sequence length
            # then reshape the data to have the sequence length as the third dimension
            episode_length = data_array.shape[1]
            data_array = data_array[:, :(episode_length - episode_length % self.sequence_length), :].reshape(-1, self.sequence_length, data_array.shape[2])

            # use zero point as first point if need
            if use_zero_point:
                for idx in range(data_array.shape[0]):
                    data = data_array[idx,:,:]
                    if np.isclose(data[0,0], 0, atol=1) and np.isclose(data[0,1], 0, atol=1):
                        continue
                    else:
                        first_column = data[0, :2]
                        data[:, :2] -= first_column[np.newaxis, :]

            return torch.tensor(data_array)

        if type(path) == str:
            pickle_files = glob.glob(os.path.join(path, '*.pkl'))
        elif type(path) == list:
            pickle_files = path
        else:
            raise ValueError('Path should be a string or a list of strings')
        if len(pickle_files) == 0:
            raise ValueError(f'No pickle files found in the directory: {path}')
        print(f'Loading {len(pickle_files)} pickle files')

        self.sequence_length = history_length + action_length
        
        # DEBUG = True
        DEBUG = False
        
        if DEBUG:
            self.data = []
            for pickle_file in pickle_files:
                self.data.append(load_pickle(pickle_file))
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.data = list(tqdm.tqdm(executor.map(load_pickle, pickle_files), total=len(pickle_files)))

        # concatenate all the episodes
        self.data = torch.concatenate(self.data, axis=0)
        self.data = self.data.to(torch.float32)

        #filter out high throttle data
        if filter:
            ##
            
            filtered_idx = []
            print("prefilted data shape:", self.data.shape)
            for i in range(self.data.shape[0]):
                # check if the car is on a straight away
                throttle = self.data[i, :, -3].numpy()
                vx = self.data[i, :, 3].numpy()
                vy = self.data[i, :, 4].numpy()

                if not (abs(vy/vx).max() > 1/3 and abs(vx).min() > 0.5):
                    filtered_idx.append(i)
                # if np.mean(throttle) < 0.8:
                    # filtered_idx.append(i)

            self.data = self.data[filtered_idx, :, :]
            print("filtered data shape:", self.data.shape)

        self.len = self.data.shape[0]

        # create a delta dataset
        self.delta_data = self.data.clone().detach()
        
        #self.delta_data[:, 1:, :3] = self.data[:, 1:, :3] - self.data[:, :-1, :3]      #  x, y , yaw, is delta value 
        # set x, y , yaw, vx, vy, yawrate all is delta value!!!! 
        self.delta_data[:, 1:, :6] = self.data[:, 1:, :6] - self.data[:, :-1, :6]
        
        self.delta_data[:, 1:, 2] = align_yaw(self.delta_data[:, 1:, 2], 0.0)
        original_yaw = self.data[:, :-1, 2]
        # transform to Body Frame
        delta_x = self.delta_data[:, 1:, 0] * torch.cos(original_yaw) + self.delta_data[:, 1:, 1] * torch.sin(original_yaw)   # x transfor to vehicie body frame
        delta_y = -self.delta_data[:, 1:, 0] * torch.sin(original_yaw) + self.delta_data[:, 1:, 1] * torch.cos(original_yaw)  # y transfor to vehicie body frame

        self.delta_data[:, 1:, 0] = delta_x
        self.delta_data[:, 1:, 1] = delta_y
        self.delta_data = self.delta_data.detach()

        # split the delta data into history, action and future
        self.history = self.delta_data[:, :history_length, :8]
        self.action = self.delta_data[:, history_length-1:history_length+action_length-1, 6:8]
        self.y = self.delta_data[:, history_length:history_length+action_length, :6]

        # get the mean and std of the data
        if mean is not None:
            self.mean = mean
        else:
            self.mean = torch.mean(self.delta_data[:, 1:, :6], axis=(0, 1))
        if std is not None:
            self.std = std
        else:
            self.std = torch.std(self.delta_data[:, 1:, :6], axis=(0, 1))

        
        if self.attack:
            # add random noise to history
            noise_locations = torch.randint(0, history_length, (len(self.history),))
            self.history[range(len(self.history)), noise_locations, 3] += torch.rand(len(self.history)) * 60 - 30
            
        if self.add_noise:
            self.history[:, :, 0] += torch.rand_like(self.history[:, :, 0]) * 0.01 - 0.005
            self.history[:, :, 1] += torch.rand_like(self.history[:, :, 1]) * 0.01 - 0.005
            self.history[:, :, 2] += torch.rand_like(self.history[:, :, 2]) * 0.01 - 0.005
            self.history[:, :, 3] += torch.rand_like(self.history[:, :, 3]) * 1. - 0.5
            self.history[:, :, 4] += torch.rand_like(self.history[:, :, 4]) * 0.1 - 0.05
            self.history[:, :, 5] += torch.rand_like(self.history[:, :, 5]) * 0.01 - 0.005
            
            ## actions
            self.history[:, :, 6] += torch.rand_like(self.history[:, :, 6]) * 0.05 - 0.025
            self.history[:, :, 7] += torch.rand_like(self.history[:, :, 7]) * 0.05 - 0.025


        # expand the dataset for teacher forcing
        if teacher_forcing:
            if binary_mask:
                self.action, self.action_padding_mask = generate_subsequences_hf(self.action)
            else:
                self.action, self.action_padding_mask = generate_subsequences(self.action)
            self.history = torch.repeat_interleave(self.history, self.action.shape[1], dim=0)
            self.y = torch.repeat_interleave(self.y, self.action.shape[1], dim=0)
            self.len = self.history.shape[0]
            # self.data = torch.cat([self.history, 
            #                        torch.cat(
            #                            [self.y, torch.cat(
            #                                [self.action[:, 1:, :], torch.zeros_like(self.action[:, 0:1, :])], axis=1
            #                            )], axis=2)
            #                       ], axis=1)
            self.data = torch.repeat_interleave(self.data, self.action.shape[1], dim=0)
        else:
            if binary_mask:
                self.action_padding_mask = torch.ones(self.action.shape[0], self.action.shape[1])
            else:
                self.action_padding_mask = torch.zeros(self.action.shape[0], self.action.shape[1])
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        action_padding_mask = None if self.action_padding_mask is None else self.action_padding_mask[idx]
        return self.history[idx], self.action[idx], self.y[idx], action_padding_mask #, self.data[idx]
    
    def get_episode(self, idx):
        return self.data[idx]

    def get_total_data(self):
        return self.data
    
    def remove_episode(self, id_list):

        mask = torch.ones(self.history.size(0), dtype=bool)
        mask[id_list] = False
        
        self.data = self.data[mask]
        self.history = self.history[mask]
        self.action = self.action[mask]
        self.y = self.y[mask]
        self.action_padding_mask = self.action_padding_mask[mask]

        self.len = len(self.data)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, flip=False):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(0.1)
        self.flip = flip

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.flip:
            x = x + torch.flip(self.pe[:x.size(1)], [0])
        else:
            x = x + self.pe[:x.size(1)]

        return self.dropout(x)

class TorchTransformerDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads, num_layers, device, dropout=0.1, history_length=250, prediction_length=50):

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.prediction_length = prediction_length

        self.odd_indices = torch.arange(0, history_length * 2 - 1, 2, device=device)
        self.even_indices = torch.arange(1, history_length * 2 - 1, 2, device=device)

        self.state_embedding = nn.Linear(state_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.output_embedding = nn.Linear(latent_dim, self.output_dim)

        self.history_pos_emb = LearnedPositionalEncoding(latent_dim, history_length * 2 - 1, flip=True)
        self.action_pos_emb = LearnedPositionalEncoding(latent_dim, prediction_length)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None):
        # history_emb = torch.zeros(history.size(0), history.size(1) * 2 - 1, self.latent_dim, device=history.device)
        # history_emb[self.odd_indices] = self.state_embedding(history[:, :, :self.state_dim]) # shape: [batch_size, seq_length, latent_dim]
        # history_emb[self.even_indices] = self.action_embedding(history[:, :-1, self.state_dim:self.state_dim+self.action_dim]) # shape: [batch_size, seq_length-1, latent_dim]
        state_emb = self.state_embedding(history[:, :, :self.state_dim])
        action_emb = self.action_embedding(history[:, :, self.state_dim:self.state_dim+self.action_dim])
        history_emb = torch.cat((state_emb[:, None, :, :], action_emb[:, None, :, :]), dim=1).view(-1, 2 * self.history_length, self.latent_dim).transpose(1, 2).contiguous().view(-1, 2 * self.history_length, self.latent_dim)[:, :-1, :]

        history_emb = self.history_pos_emb(history_emb)

        action_emb = self.action_embedding(action)
        action_emb = self.action_pos_emb(action_emb)

        x = self.transformer_decoder(action_emb, history_emb,
                                     tgt_is_causal=True, # memory_is_causal=True,
                                     tgt_mask = nn.Transformer.generate_square_subsequent_mask(action_emb.size(1), action_emb.device),
                                    #  memory_mask = nn.Transformer.generate_square_subsequent_mask(history_emb.size(1), device=history_emb.device),
                                     tgt_key_padding_mask=action_padding_mask,
                                     memory_key_padding_mask=history_padding_mask
                                    )
        x = self.output_embedding(x)
        return x