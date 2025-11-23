import numpy as np
import matplotlib as plot
import torch

from data import encode
from data import decode
from data import get_batch

import config

xb, yb = get_batch("train")

# how the transformer looks ahead
for b in range(config.batch_size):
    for i in range(config.block_size):
        context = xb[b, :i+1] # 2d array accessing
        target = yb[b, i]
        # print(f"When we have {context} the next is {target}")

"""
Attention = softmax(QK^T/sqrt(d_k))
Output = Attention * V
"""

dk = config.batch_size
print(dk)