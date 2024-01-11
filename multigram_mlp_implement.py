import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple, Dict, List
from tqdm import tqdm

SEP = '#'

class MLP(torch.nn.Module):
    def __init__(self, n_gram: int = 3, embed_dim: int = 10, hidden_layer_dim: int = 1000):
        super().__init__()
        # update model meta
        self._total_ch = 27
        self._n_gram = n_gram
        self._embed_dim = embed_dim
        self._hidden_layer_dim = hidden_layer_dim
        # embedding
        self._embed = torch.nn.Embedding(num_embeddings=self._total_ch, embedding_dim=embed_dim)
        # layer1
        self._linear1 = torch.nn.Linear(in_features=self._n_gram * embed_dim, out_features=hidden_layer_dim)
        self._bn1 = torch.nn.BatchNorm1d(num_features=hidden_layer_dim)
        self._act1 = torch.nn.ReLU()
        # layer2
        self._linear2 = torch.nn.Linear(in_features=self._hidden_layer_dim, out_features=self._total_ch)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # embedding
        out = self._embed(input_tensor)
        out = out.reshape((out.shape[0], -1))
        # layer1
        out = self._linear1(out)
        out = self._bn1(out)
        out = self._act1(out)
        # layer2
        out = self._linear2(out)
        return out

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
    train_matrix, train_label = [], []
    for word in words:
        word = (SEP * n_gram) + word + SEP
        # build n-gram dataset item
        for idx in range(n_gram, len(word)):
            input_word = word[idx-n_gram: idx]
            output_character = word[idx]
            # build train item
            train_matrix.append([ch_to_idx_map[ch] for ch in input_word])
            train_label.append(ch_to_idx_map[output_character])
    # build tensor
    train_matrix_torch = torch.from_numpy(np.asarray(train_matrix))
    train_label_torch = torch.from_numpy(np.asarray(train_label))
    return ch_to_idx_map, idx_to_ch_map, train_matrix_torch, train_label_torch

# loss
def loss(model_output: torch.Tensor, gt_output: torch.Tensor) -> torch.Tensor:
    model_output_norm = model_output.exp()
    model_output_softmax = model_output_norm / model_output_norm.sum(dim=1, keepdim=True)
    logits = model_output_softmax[torch.arange(gt_output.shape[0]), gt_output]
    scalar_loss = - logits.log().mean()
    return scalar_loss

# training config
n_gram = 3
lr = 0.001
lr_decay_rate = 0.99
lr_decay_iter = 10000
log_iter_interval = 1000
dataset_limit = -1
total_epoch, batch_size = 10, 32

# build model & optimizer
model = MLP(n_gram=n_gram)

# build dataset
words = load_words('./names.txt')
ch_to_idx_map, idx_to_ch_map, train_matrix_torch, train_label_torch = build_dataset(words=words, n_gram=n_gram)
train_matrix_torch = train_matrix_torch[: dataset_limit] if dataset_limit > 0 else train_matrix_torch
train_label_torch = train_label_torch[: dataset_limit] if dataset_limit > 0 else train_label_torch

# Training loop
generator = torch.Generator()
cur_iter = 0
for epoch in range(1, total_epoch+1):
    # shuffle data indexes
    generator.manual_seed(epoch)
    data_indexes = torch.randperm(train_matrix_torch.shape[0], generator=generator)
    for iter, batch_indexes in tqdm(enumerate(torch.split(data_indexes, split_size_or_sections=batch_size))):
        train_sample = train_matrix_torch[batch_indexes]
        train_sample_label = train_label_torch[batch_indexes]
        # forward model and update params based on grad
        model_output = model(train_sample)
        scalar_loss = loss(model_output=model_output, gt_output=train_sample_label)
        loss_value = scalar_loss.item()
        scalar_loss.backward()
        for param in model.parameters():
            if param.grad is not None and param.requires_grad:
                param.data -= lr * param.grad
        model.zero_grad()
        if iter % log_iter_interval == 0:
            print(f'epoch: {epoch}, iter: {iter}, loss: {scalar_loss}, lr: {lr}')
        cur_iter += 1
        if cur_iter % lr_decay_iter == 0:
            lr = lr * lr_decay_rate
