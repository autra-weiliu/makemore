'''
bigram probability based word generator, non-nn solution
'''
import torch

from typing import Tuple, Dict, List


def load_words(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        return f.read().splitlines()


def build_bigram_counting_map(words: List[str]) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int], Dict[int, str]]:
    ''' bigram pair counting map + char to index map + index to char map '''
    all_chars = sorted(set([c for word in words for c in word] + ['#']))
    # char to index map and index to char map
    ch_to_idx_map, idx_to_ch_map = {}, {}
    for idx, ch in enumerate(all_chars):
        ch_to_idx_map[ch] = idx
        idx_to_ch_map[idx] = ch
    # build bigram dict
    bigram_counting_map = {}
    for word in words:
        word = '#' + word + '#'
        for ch1, ch2 in zip(word, word[1: ]):
            pair = (ch_to_idx_map[ch1], ch_to_idx_map[ch2])
            bigram_counting_map[pair] = bigram_counting_map.get(pair, 0) + 1
    return bigram_counting_map, ch_to_idx_map, idx_to_ch_map


def build_tensor(bigram_counting_map: Dict[Tuple[int, int], int], n: int) -> torch.tensor:
    bigram_tensor = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            count = bigram_counting_map.get((i, j), 0)
            bigram_tensor[i, j] = count
    return bigram_tensor


words = load_words('./names.txt')
bigram_counting_map, ch_to_idx_map, idx_to_ch_map = build_bigram_counting_map(words=words)
bigram_tensor = build_tensor(bigram_counting_map=bigram_counting_map, n=len(ch_to_idx_map))
p_tensor = bigram_tensor / bigram_tensor.sum(dim=1, keepdim=True)

# TODO(hack tensor to make it uniform)
# p_tensor = torch.ones_like(p_tensor) / 27.0

generator = torch.Generator()
generator.manual_seed(0)
total = 10

for index in range(total):
    current_idx, end_idx = ch_to_idx_map['#'], ch_to_idx_map['#']
    result_list = []
    while True:
        next_idx = torch.multinomial(p_tensor[current_idx], num_samples=1, generator=generator).item()
        if next_idx == end_idx:
            break
        else:
            result_list.append(idx_to_ch_map[next_idx])
            current_idx = next_idx
    print(f'word {index} is {"".join(result_list)}')
