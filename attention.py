import matplotlib.pyplot as plt
import numpy as np
import torch

# opening the text file
file_path = "data/shakespeare.txt"

input_file = open(file_path, "r")

text = input_file.read()

input_file.close() # must include this close otherwise there will be a exit error

# print(len(text)) # 1115394

char_list = sorted(list(set(text))) # gets all possible characters seen in text
print("".join(char_list)) # first character is a \n and second is a space

# encoder (Assign each character to a number)


# decoder