import random


def swap_token(tokens,p = 0.3):
    num_swaps = max(1, int(p * len(tokens)))
    if len(tokens) > num_swaps + 2:
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    return tokens

def dele_tokens(tokens,p = 0.3):
    num_deletes = max(1, int(p * len(tokens)))
    if len(tokens) > num_deletes + 2:
        for _ in range(num_deletes):
            idx = random.randint(0, len(tokens) - 1)
            tokens.pop(idx)
    return tokens

def insert_punctuations(tokens,p = 0.3):
    num_inserts = max(1, int(p * len(tokens)))
    punctuations = ['。', ',', '!', '?']
    if len(tokens) > num_inserts + 2:
        for _ in range(num_inserts):
            idx = random.randint(0, len(tokens) - 1)
            punctuation = random.choice(punctuations)
            tokens.insert(idx, punctuation)
    return tokens