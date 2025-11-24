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

    d_k = 32
    attention = ((Q @ K.t()) / d_k).softmax(1)

    return attention @ V

Q = torch.rand(2,2)
K = torch.rand(2,2)
V = torch.rand(2,2)

print(Q)
print(K)
print(V)

print(scaled_dot_product_attention(Q,K,V))