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
# print("".join(char_list)) # first character is a \n and second is a space

# encoder (Assign each character to a number)
# enum = dict(enumerate(char_list)) # dictionary of all characters mapped to number from 0-65
# print(enum)

def encode(tokens):
    encoding = []
    for char in tokens:
        encoding.append(char_list.index(char))
    
    return encoding

print(encode("point"))

# decoder

def decode(vector):
    string = ""
    for item in vector:
        string = string + char_list[item]
    
    return string

print(decode(encode("point")))