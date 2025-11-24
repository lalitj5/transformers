import numpy as np
import matplotlib as plot
import torch

from data import encode, decode, get_batch

import config
"""
xb, yb = get_batch("train")

# how the transformer looks ahead
for b in range(config.batch_size):
    for i in range(config.block_size):
        context = xb[b, :i+1] # 2d array accessing
        target = yb[b, i]
        # print(f"When we have {context} the next is {target}")
"""

"""
Attention = softmax(QK^T/sqrt(d_k))
Output = Attention * V
"""
def scaled_dot_product_attention(Q, K, V, mask = None):
    # Q, K, V are tensors
    # d_k will be specified
    d_k = K.shape[-1]
    attention = ((Q @ K.transpose(-2,-1)) / (d_k ** 0.5)).softmax(-1)

    # replace the values with -inf as softmax -inf = 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    return attention @ V

Q = torch.rand(4, 8, 64)
K = torch.rand(4, 8, 64)
V = torch.rand(4, 8, 64)
print(scaled_dot_product_attention(Q,K,V))