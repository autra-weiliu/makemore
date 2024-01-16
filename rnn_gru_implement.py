
import torch
import torch.nn.functional as F
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

def evaluation(model: torch.nn.Module, eval_matrix: torch.Tensor, eval_label: torch.Tensor, device: torch.device, eval_log_interval=1000):
    acc, total, eval_loss = 0.0, eval_matrix.shape[0], 0.0
    with torch.no_grad():
        for sample_id in tqdm(range(total)):
            data, label = eval_matrix[sample_id].to(device, non_blocking=True), eval_label[sample_id].to(device, non_blocking=True)
            data = torch.unsqueeze(data, dim=0)
            model_output = model(data)
            # get acc
            pred_output = torch.squeeze(model_output[0, -1, :])
            output_label = torch.argmax(pred_output)
            if output_label == label:
                acc += 1
            # get loss
            rnn_labels = torch.cat([data[0, 1:], label.unsqueeze(0)], dim=0)
            loss = F.cross_entropy(model_output[0, :, :].view(-1, model_output.shape[-1]), rnn_labels.view(-1))
            eval_loss += loss
            if sample_id % eval_log_interval == 0:
                print(f'{sample_id + 1} samples\' acc: {acc / (sample_id + 1)}, loss: {eval_loss / (sample_id + 1)}')
    print(f'final acc rate: {acc / total}, loss: {eval_loss / total}')

class RnnCell(torch.nn.Module):
    def __init__(self, word_embed_dim: int, hidden_embed_dim: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(in_features=word_embed_dim + hidden_embed_dim, out_features=hidden_embed_dim)

    def forward(self, x: torch.Tensor, hprev: torch.Tensor) -> torch.Tensor:
        out = self._linear(torch.cat([x, hprev], dim=1))
        out = F.tanh(out)
        return out

class GruCell(torch.nn.Module):
    def __init__(self, word_embed_dim: int, hidden_embed_dim: int) -> None:
        super().__init__()
        # TODO implement it

    def forward(self, x: torch.Tensor, hprev: torch.Tensor) -> torch.Tensor:
        pass

class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 27, word_embed_dim: int = 64, hidden_embed_dim: int = 64) -> None:
        super().__init__()
        # dims
        self._word_embed_dim = word_embed_dim
        self._hidden_embed_dim = hidden_embed_dim
        self._vocab_size = vocab_size
        # modules
        self._vocab_embed = torch.nn.Embedding(num_embeddings=self._vocab_size, embedding_dim=self._word_embed_dim)
        self._init_state = torch.nn.Parameter(torch.zeros(1, self._hidden_embed_dim))
        self._lm_head = torch.nn.Linear(in_features=self._hidden_embed_dim, out_features=self._vocab_size)
        self._cell = RnnCell(word_embed_dim=self._word_embed_dim, hidden_embed_dim=self._hidden_embed_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_embed = self._vocab_embed(input)
        # B, T, word_dim
        b, t, _ = input_embed.shape
        hprev = self._init_state.expand((b, self._hidden_embed_dim))
        # gather h outputs
        hidden_outputs = []
        for timestamp in range(t):
            x = input_embed[: , timestamp, :]
            # B, word_dim and B, hidden_dim
            next_h = self._cell(x, hprev)
            hprev = next_h
            hidden_outputs.append(hprev)
        # hidden output to final output
        # B, T, hidden_dim
        return self._lm_head(torch.stack(hidden_outputs, dim=1))

# training config
seed = 0
n_gram = 10
embed_dim = 16
lr = 0.01
lr_decay_rate = 0.9
lr_decay_iter = 2000
log_iter_interval = 1000
dataset_limit = -1
total_epoch, batch_size = 6, 32

# build model & optimizer
device = torch.device(0)
model = RNNModel(word_embed_dim=128, hidden_embed_dim=128)
model.train().to(device=device)
model.zero_grad()

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
        model_output = model(train_sample)
        # concat train's label and next labels
        rnn_labels = torch.cat([train_sample[:, 1:], train_sample_label.unsqueeze(1)], dim=1)
        # loss
        loss = F.cross_entropy(model_output.view(- 1, model_output.shape[-1]), rnn_labels.view(-1))
        loss_value = loss.item()
        all_losses.append(loss_value)
        # optimizer
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None and param.requires_grad:
                    param.data -= lr * param.grad
        model.zero_grad()
        cur_iter += 1
        if cur_iter % log_iter_interval == 0:
            print(f'epoch: {epoch}, iter: {cur_iter}, lr: {lr} train loss: {loss_value}')
        if cur_iter % lr_decay_iter == 0:
            lr *= lr_decay_rate

# evaluation
model.eval()
evaluation(model=model, eval_matrix=eval_matrix_torch, eval_label=eval_label_torch, device=device)
