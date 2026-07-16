import torch
from torch import nn
import ast
import stoi
import itos
import input_target
import random

'''
1. Make a batch function that returns x and y tensors.
2. Build a bigram model that maps token IDs to logits.
3. Run a forward pass and verify logits shape.
4. Compute cross-entropy loss.
5. Train for several iterations.
6. Verify loss decreases.
7. Generate a small text sample.
'''

class DirectBigramLookupModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.embedding.parameters(), lr= 0.1)

    def forward(self, x, target=None):
        logits = self.embedding(x)

        if target == None:
            return logits
        else:
            prediction = torch.reshape(logits, [32,32])
            targets = torch.reshape(target, [32])
            loss = self.criterion(prediction, targets)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            return logits, loss.item()

class EmbeddingProjectionBigramModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer = torch.optim.SGD(list(self.embedding.parameters())+list(self.linear.parameters()), lr= 0.1)

    def forward(self, x, target=None):
        emb = self.embedding(x)
        logits = self.linear(emb)

        if target == None:
            return logits
        else:
            prediction = torch.reshape(logits, [32,32])
            targets = torch.reshape(target, [32])
            loss = self.criterion(prediction, targets)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            return logits, loss.item()

'''
1. Store a trainable token-to-logits table.
2. Receive x_tensor with shape [B, T].
3. Return logits with shape [B, T, V].
4. Optionally compute loss when y_tensor is provided.
5. Later, generate text by repeatedly predicting the next token.
    torch.nn
'''

def train_val_data(text, train_data_percentage):
    n_train_data = int(len(text)*train_data_percentage)
    train_data = text[0:n_train_data]
    validation_data = text[n_train_data:]
    return train_data, validation_data

def get_batch(text, block_size, batch_size, indices=None):
    x = []
    y = []
    upper_limit = len(text)-block_size
    if upper_limit < 0:
        print("Text too short for block size")
        exit()

    
    if indices == None:
        ind = random.choices(range(len(text)-block_size), k=batch_size)
    else:
        ind = indices

    if max(ind) > upper_limit:
        print("Indeces out of range")
        exit()

    print(ind)
    for i in ind:
        if((i+block_size+1)>len(text)):
            print("Text too short for B*T size")
            break
        x.append(text[i:i+block_size])
        y.append(text[i+1:i+block_size+1])

    return x, y

def train_model(model, id_input, id_target, n_lr_steps=1000):
    #ID to tensor form
    x_tensor = torch.tensor(id_input, dtype=torch.long)
    y_tensor = torch.tensor(id_target, dtype=torch.long)

    loss = []
    print("\nTraining...")
    for x in range(n_lr_steps):
        logits, loss_v = model(x_tensor, y_tensor)
        loss.append(loss_v)

    print(f"Loss for {n_lr_steps} steps: Min= {min(loss)} Max= {max(loss)} Avg= {sum(loss)/len(loss)}")
    return

def generate(model, id_input, norm="softmax", steps=10):
    
    print("\nGenerate...")
    with torch.no_grad():
        output_token_id = [id_input]
        for y in range(steps):
            logits = model(torch.tensor([output_token_id[-1]], dtype=torch.long))
            argmax_token_id = torch.argmax(logits).item()
            probabilities = torch.softmax(logits, dim=1)
            softmax_token_id = torch.multinomial(probabilities, num_samples=1).item()

            if norm == "softmax":
                output_token_id.append(softmax_token_id)
            else:
                output_token_id.append(argmax_token_id)

        return output_token_id
        
        

if __name__ == "__main__":


    block_size = 8 #int(input("Block Size: "))
    batch_size = 4 #int(input("Batch Size: "))
    vocab = [' ', ',', '-', '.', '/', 'D', 'I', 'T', 'W', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'] #ast.literal_eval(input("Vocabulary sorted: "))
    text =  "This is an example text, to train a character-level transformer. We expect to set a vocabulary, test text to IDs and IDs to text logic using stoi and itos, then after testing it works, we can create input/target pairs, and batches of these pairs to train the model." #input("Text: ")
    
#Train and Validation Data

    tr_data, val_data = train_val_data(text, 0.9)

    print(tr_data)
    print(val_data)
    

#Random Batches of data

    x, y = get_batch(tr_data, block_size, batch_size)
    
    print(x)
    print(y)

    #String to ID

    id_input = stoi.string_to_ids(vocab, x)
    id_target = stoi.string_to_ids(vocab, y)

    print(id_input)
    print(id_target)
    
#Init model

    vocab_n = 32
    n_embd = 16

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    bigram_model = EmbeddingProjectionBigramModel(vocab_n, n_embd).to(device)
    

#Training

    train_model(bigram_model, id_input, id_target)

#Generate

    id_input =  stoi.string_to_ids(vocab, "This")

    id_output = generate(bigram_model, id_input[0][-1], "softmax")

    print(id_output)

    output_token = itos.ids_to_string(vocab, id_output)

    print(output_token)