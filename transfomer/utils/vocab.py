from collections import Counter
import re
import torch

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, reserved_tokens=None, min_freq=3):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频并排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + ['<padding>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(idx) for idx in indices]
    
    def to_idx(self, indices):
        if not isinstance(indices, (list, tuple)):
            if not indices in self.token_to_idx:
                return self.unk
            else:
                return self.token_to_idx[indices]
        return [self.to_idx(idx) for idx in indices]

    @property
    def unk(self):
        return 0		# 未知token索引为0
    
    @property
    def padding(self):
        return 1		# padding索引为1
    
    # @property
    # def token_freqs(self):
    #     return self._token_freqs





