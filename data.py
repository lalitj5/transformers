import torch
import config

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

# creating the tensors of multiple words
data = torch.tensor(encode(text), dtype=torch.long) 
# torch.long not needed for simple encoding as range is 0-65, but for scaled versions it will be necessary

# determining training and validation sets
percentage = 0.9 # can change this at will
threshold = int(len(data) * percentage)

training = data[:threshold]
validation = data[threshold:]

# training[:block_size+1] this is a piece of the training set; brings out 9 characters, but has 8 chances ot predict
torch.manual_seed(1337)
def get_batch(split):
    if split == "train":
        data = training
    else:
        data = validation
    # grab random positions (low is optional so this parameter is the high)
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # first time using vectorized form of iteration, seems to be recommended for pytorch/tensor apps
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])

    return x, y