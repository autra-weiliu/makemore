import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Dict, List
from tqdm import tqdm

SEP = '#'

class LinearBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self._bn = torch.nn.BatchNorm1d(num_features=out_features)
        self._relu = torch.nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out = self._linear(input_tensor)
        out = self._bn(out)
        out = self._relu(out)
        return out

# NOTE: nn.module is more complicated then expected
class MLP(torch.nn.Module):
    def __init__(
        self,
        n_gram: int = 3,
        input_dim: int = 27,
        embed_dim: int = 16,
        hidden_layer_dims: Tuple[int] = (200, 400, 800, 400),
        output_dim: int = 27
    ):
        # NOTE: important!!
        super().__init__()
        # update model meta
        self._n_gram = n_gram
        self._embed_dim = embed_dim
        # embedding
        self._embed = torch.nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim)
        # hidden layers block
        assert len(hidden_layer_dims) > 0, 'hidden layer should not be empty'
        self._blocks = torch.nn.Sequential()
        for idx, hidden_layer_dim in enumerate(hidden_layer_dims):
            # two dims
            in_features = self._n_gram * embed_dim if idx == 0 else hidden_layer_dims[idx - 1]
            out_features = hidden_layer_dim
            # append block
            self._blocks.append(LinearBlock(in_features=in_features, out_features=out_features))
        # classification layer
        self._linear = torch.nn.Linear(in_features=hidden_layer_dims[-1], out_features=output_dim)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # embedding
        out = self._embed(input_tensor)
        # flatten multi gram's embedding features
        out = out.reshape((out.shape[0], -1))
        # hidden layers
        out = self._blocks(out)
        # classification layer
        out = self._linear(out)
        return out

class HierachyLinearBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self._bn = torch.nn.BatchNorm1d(num_features=out_features)
        self._relu = torch.nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim == 3:
            # for bn, need to permute dims
            out = self._linear(input_tensor).permute(0, 2, 1)
            out = self._bn(out)
            out = out.permute(0, 2, 1)
            out = self._relu(out)
        else:
            out = self._linear(input_tensor)
            out = self._bn(out)
            out = self._relu(out)
        return out

class HierachyMLP(torch.nn.Module):
    def __init__(
        self,
        n_gram: int = 3,
        input_dim: int = 27,
        embed_dim: int = 16,
        hidden_layer_dim: int = 512,
        output_dim: int = 27
    ):
        # NOTE: important!!
        super().__init__()
        # update model meta
        assert n_gram - (n_gram & (- n_gram)) == 0, 'n_gram should be power of two'
        self._n_gram = n_gram
        self._embed_dim = embed_dim

        # embedding
        self._embed = torch.nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim)

        # middle blocks
        self._blocks = torch.nn.ModuleList()
        # first block
        self._blocks.append(HierachyLinearBlock(in_features=embed_dim * 2, out_features=hidden_layer_dim))
        # middle blocks
        layer_value = n_gram
        while layer_value > 2:
            self._blocks.append(HierachyLinearBlock(in_features=hidden_layer_dim * 2, out_features=hidden_layer_dim))
            layer_value /= 2
        # last block
        self._blocks.append(HierachyLinearBlock(in_features=hidden_layer_dim, out_features=hidden_layer_dim))

        # classification layer
        self._linear = torch.nn.Linear(in_features=hidden_layer_dim, out_features=output_dim)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # embedding
        out = self._embed(input_tensor)
        # hidden layers
        out = out.view(out.shape[0], -1, out.shape[-1] * 2)
        out = self._blocks[0](out)
        for block_idx in range(1, len(self._blocks) - 1):
            out = out.contiguous().view(out.shape[0], -1, out.shape[-1] * 2)
            out = self._blocks[block_idx](out)
        out = out.squeeze(1)
        out = self._blocks[-1](out)
        # classification layer
        out = self._linear(out)
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

# loss
def loss(model_output: torch.Tensor, gt_output: torch.Tensor) -> torch.Tensor:
    assert model_output.shape[: 1] == gt_output.shape, f'model output shape: {model_output.shape} and gt output shape: {gt_output.shape} is not aligned'
    model_output_norm = model_output.exp()
    model_output_softmax = model_output_norm / model_output_norm.sum(dim=1, keepdim=True)
    # model_output_softmax = F.softmax(model_output, dim=-1)
    assert abs(model_output_softmax[0].sum().item() - 1) < 1e-4, f'model output softmax sum should be one but {model_output_softmax[0].sum().item()} is returned'
    logits = model_output_softmax[torch.arange(gt_output.shape[0]), gt_output]
    scalar_loss = - logits.log().mean()
    return scalar_loss

def evaluation(model: torch.nn.Module, eval_matrix: torch.Tensor, eval_label: torch.Tensor, device: torch.device, eval_log_interval=1000):
    acc, total, eval_loss = 0.0, eval_matrix.shape[0], 0.0
    with torch.no_grad():
        for sample_id in tqdm(range(total)):
            data, label = eval_matrix[sample_id].to(device, non_blocking=True), eval_label[sample_id].to(device, non_blocking=True)
            data = torch.unsqueeze(data, dim=0)
            output = model(data)
            output_label = torch.argmax(torch.squeeze(output, dim=0))
            if output_label == label:
                acc += 1
            softmax_output = F.softmax(output, dim=-1)
            eval_loss += (- softmax_output[0, label].log())
            if sample_id % eval_log_interval == 0:
                print(f'{sample_id + 1} samples\' acc: {acc / (sample_id + 1)}, loss: {eval_loss / (sample_id + 1)}')
    print(f'final acc rate: {acc / total}, loss: {eval_loss / total}')

# training config
seed = 0
n_gram = 32
embed_dim = 64
lr = 0.01
lr_decay_rate = 0.9
lr_decay_iter = 2000
log_iter_interval = 1000
dataset_limit = -1
total_epoch, batch_size = 20, 32

# build model & optimizer
device = torch.device(0)
model = HierachyMLP(n_gram=n_gram, embed_dim=embed_dim)
model.to(device)
model.train()

print('----------------------- parameters -----------------------')
total_params = 0
for name, param in model.named_parameters():
    print(name, param.shape)
    total_params += param.numel()
print(f'total number of parameters: {total_params}')
print('----------------------------------------------------------')
print('------------------------- buffers ------------------------')
for name, buffer in model.named_buffers():
    print(name, buffer.shape)
print('----------------------------------------------------------')

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
        # forward model and update params based on grad
        model_output = model(train_sample)
        scalar_loss = loss(model_output=model_output, gt_output=train_sample_label)
        loss_value = scalar_loss.item()
        all_losses.append(loss_value)
        scalar_loss.backward()
        # optimizer optimize parameters
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None and param.requires_grad:
                    param.data -= lr * param.grad
        model.zero_grad()
        if iter % log_iter_interval == 0:
            print(f'epoch: {epoch}, iter: {iter}, loss: {loss_value}, lr: {lr}')
        cur_iter += 1
        if cur_iter % lr_decay_iter == 0:
            lr = lr * lr_decay_rate

# eval model
model.eval()
evaluation(model=model, eval_matrix=eval_matrix_torch, eval_label=eval_label_torch, device=device)

# visualize all the losses
loss_batch_size = 1000
losses_length = len(all_losses) // loss_batch_size * loss_batch_size
losses_tensor = torch.asarray(all_losses[: losses_length])
plt.plot(losses_tensor.view(-1, loss_batch_size).mean(dim=1))
plt.show()

# model generation example
model.eval()
context = [ch_to_idx_map[SEP]] * n_gram
output_string = ''
while True:
    input_tensor = torch.unsqueeze(torch.asarray(context), dim=0)
    input_tensor = input_tensor.to(device, non_blocking=True)
    output_tensor = model(input_tensor)
    output_idx = torch.argmax(output_tensor[0, :], dim=0).item()
    if output_idx == ch_to_idx_map[SEP]:
        break
    else:
        output_string += idx_to_ch_map[output_idx]
        context = context[1: ] + [output_idx]
print(output_string)
