import matplotlib.pyplot as plt
import numpy as np
import torch

# opening the text file
file_path = "data/shakespeare.txt"

input_file = open(file_path, "r")

text = input_file.read()
# print(len(text)) # 1115394

input_file.close() # must include this close otherwise there will be a exit error

char_list = sorted(list(set(text))) # gets all possible characters seen in text
# print("".join(char_list)) # first character is a \n and second is a space

# encoder (Assign each character to a number)
def encode(tokens):
    encoding = []
    for char in tokens:
        encoding.append(char_list.index(char))
    
    return encoding


# decoder (reverse encoder)
def decode(vector):
    string = ""
    for item in vector:
        string = string + char_list[item]
    
    return string

# debug statement to make sure these functions work
# print(encode("hii there"))
# print(decode(encode("hii there")))

# creating the tensors of multiple words
data = torch.tensor(encode(text), dtype=torch.long) 
# torch.long not needed for simple encoding as range is 0-65, but for scaled versions it will be necessary
# print(data[:1000])

# determining training and validation sets
percentage = 0.9 # can change this at will
threshold = int(len(data) * percentage)

training = data[:threshold]
validation = data[threshold:]

block_size = 8 # size of samples
batch_size = 4 # number of samples

# training[:block_size+1] this is a piece of the training set; brings out 9 characters, but has 8 chances ot predict
torch.manual_seed(1337)
def get_batch(split):
    if split == "train":
        data = training
    else:
        data = validation
    # grab random positions (low is optional so this parameter is the high)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # first time using vectorized form of iteration, seems to be recommended for pytorch/tensor apps
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y

# creating examples
xb, yb = get_batch("train") # xb are inputs, yb are outputs

# how the transformer looks ahead 
for b in range(batch_size):
    for i in range(block_size):
        context = xb[b, :i+1] # 2d array accessing
        target = yb[b, i]
        print(f"When we have {context} the next is {target}")