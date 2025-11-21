import matplotlib.pyplot as plt
import numpy as np
import torch

# opening the text file
file_path = "data/shakespeare.txt"

input_file = open(file_path, "r")

text_body = input_file.read()

input_file.close() # must include this close otherwise there will be a exit error



