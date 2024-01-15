
import torch
import numpy as np

from typing import Tuple, Dict, List
from tqdm import tqdm

SEP = '#'

def load_words(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return f.read().splitlines()

def build_dataset(words: List[str], n_gram: int = 3) -> Tuple[Dict[str, int], Dict[int, str]]:
    ''' bigram pair counting map + char to index map + index to char map '''
    all_chars = sorted(set([c for word in words for c in word] + [SEP]))
    # char to index map and index to char map
    ch_to_idx_map, idx_to_ch_map = {}, {}
    for idx, ch in enumerate(all_chars):
        ch_to_idx_map[ch] = idx
        idx_to_ch_map[idx] = ch
    # build dataset
    data_matrix, data_label = [], []
    for word in words:
        word = (SEP * n_gram) + word + SEP
        # build n-gram dataset item
        for idx in range(n_gram, len(word)):
            input_word = word[idx-n_gram: idx]
            output_character = word[idx]
            # build train item
            data_matrix.append([ch_to_idx_map[ch] for ch in input_word])
            data_label.append(ch_to_idx_map[output_character])
    # build tensor
    data_matrix_torch = torch.from_numpy(np.asarray(data_matrix))
    data_label_torch = torch.from_numpy(np.asarray(data_label))
    return ch_to_idx_map, idx_to_ch_map, data_matrix_torch, data_label_torch

def split_train_eval_dataset(data_matrix_torch, data_label_torch, seed, train_ratio=0.8):
    generator = torch.Generator()
    generator.manual_seed(seed)
    total_num = data_matrix_torch.shape[0]
    indexes = torch.randperm(n=total_num, generator=generator)
    train_num = int(total_num * train_ratio)
    # split train & eval dataset
    train_indexes, eval_indexes = indexes[: train_num], indexes[train_num: ]
    train_matrix, train_label = data_matrix_torch[train_indexes], data_label_torch[train_indexes]
    eval_matrix, eval_label = data_matrix_torch[eval_indexes], data_label_torch[eval_indexes]
    return train_matrix, train_label, eval_matrix, eval_label

def evaluation():
    # TODO(wei.liu) implement evaluation
    pass

# TODO(wei.liu) implement rnn model
class RNNModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(input) -> torch.Tensor:
        pass

# training config
seed = 0
n_gram = 10
embed_dim = 16
lr = 0.01
lr_decay_rate = 0.9
lr_decay_iter = 2000
log_iter_interval = 1000
dataset_limit = -1
total_epoch, batch_size = 20, 32

# build model & optimizer
device = torch.device(0)

# build dataset
words = load_words('./names.txt')
ch_to_idx_map, idx_to_ch_map, data_matrix_torch, data_label_torch = build_dataset(words=words, n_gram=n_gram)
# choose dataset subset
data_matrix_torch = data_matrix_torch if dataset_limit < 0 else data_matrix_torch[: dataset_limit]
data_label_torch = data_label_torch if dataset_limit < 0 else data_label_torch[: dataset_limit]
train_matrix_torch, train_label_torch, eval_matrix_torch, eval_label_torch = split_train_eval_dataset(data_matrix_torch=data_matrix_torch, data_label_torch=data_label_torch, seed=seed)

# Training loop
generator = torch.Generator()
cur_iter, all_losses = 0, []
for epoch in range(1, total_epoch+1):
    # shuffle data indexes
    generator.manual_seed(epoch)
    data_indexes = torch.randperm(train_matrix_torch.shape[0], generator=generator)
    for iter, batch_indexes in tqdm(enumerate(torch.split(data_indexes, split_size_or_sections=batch_size))):
        train_sample = train_matrix_torch[batch_indexes].to(device, non_blocking=True)
        train_sample_label = train_label_torch[batch_indexes].to(device, non_blocking=True)
        # TODO(wei.liu) implement model forward
