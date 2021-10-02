import numpy as np

_tok_to_idx = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4,
            'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11,
            'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 
            'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23,
            'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29,
            '-': 30, '<null_1>': 31, '<mask>': 32}


def convert(raw_batch):
    labels, sequences = list(zip(*raw_batch))

    batch_size = len(sequences)
    max_seq_len = max([len(seq) for seq in sequences])

    # round seq_len to nearest power of 2 for efficient inference
    max_seq_len = nearest_pow_2(max_seq_len)
    
    # +2 for start <cls> and end <eos> token
    tokens = np.full((batch_size, max_seq_len + 2), _tok_to_idx['<pad>'], dtype=np.int32)

    # first token should be sequence-wide <cls> token
    tokens[:, 0] = _tok_to_idx['<cls>']

    for (i, seq) in enumerate(sequences):
        tokens[i, 1:len(seq)+1] = [_tok_to_idx[tok] for tok in seq]
        tokens[i, len(seq)+1] = _tok_to_idx['<eos>']

    return labels, sequences, tokens

def nearest_pow_2(val: int) -> int:
    return int(2**np.ceil(np.log2(val)))