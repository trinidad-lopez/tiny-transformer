import torch
from torch import nn
import ast
import stoi
import itos
import input_target

'''
1. Make a batch function that returns x and y tensors.
2. Build a bigram model that maps token IDs to logits.
3. Run a forward pass and verify logits shape.
4. Compute cross-entropy loss.
5. Train for several iterations.
6. Verify loss decreases.
7. Generate a small text sample.
'''

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32, 32)
        self.embedding = nn.Embedding(32, 32)

    def forward(self, x):
        logits = self.embedding(x)
        return logits
    
'''
1. Store a trainable token-to-logits table.
2. Receive x_tensor with shape [B, T].
3. Return logits with shape [B, T, V].
4. Optionally compute loss when y_tensor is provided.
5. Later, generate text by repeatedly predicting the next token.
    torch.nn
'''

if __name__ == "__main__":
    block_size = 8 #int(input("Block Size: "))
    batch_size = 4 #int(input("Batch Size: "))
    vocab = [' ', ',', '-', '.', '/', 'D', 'I', 'T', 'W', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'] #ast.literal_eval(input("Vocabulary sorted: "))
    text =  "This is an example text, to train a character-level transformer. We expect to set a vocabulary, test text to IDs and IDs to text logic using stoi and itos, then after testing it works, we can create input/target pairs, and batches of these pairs to train the model." #input("Text: ")
    input_s, target_s = input_target.input_target_pair(block_size, batch_size, text)
    id_input = [stoi.string_to_ids(vocab, input_s[i]) for i in range(batch_size)]
    id_target = [stoi.string_to_ids(vocab, target_s[i]) for i in range(batch_size)]
    x_tensor = torch.tensor(id_input, dtype=torch.long)
    y_tensor = torch.tensor(id_target, dtype=torch.long)
    x_logit = torch.logit(x_tensor)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    bigram_model = BigramLanguageModel().to(device)
    print(bigram_model)
    print(bigram_model.named_parameters)
    print(bigram_model(x_tensor))