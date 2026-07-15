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

class DirectBigramLookupModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.layers = self.embedding
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        logits = self.layers(x)

        if target == None:
            return logits
        else:
            prediction = torch.reshape(logits, [32,32])
            targets = torch.reshape(target, [32])
            loss = self.criterion(prediction, targets)

            optimizer = torch.optim.SGD(self.layers.parameters(), lr= 0.1)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            return logits, loss

class EmbeddingProjectionBigramModel(DirectBigramLookupModel):
    def __init__(self, vocab_size, n_embd):
        super().__init__(vocab_size)
        self.emb_linear_stack = nn.Sequential(
            nn.Embedding(vocab_size, n_embd),    #Vocabulary -> Channels/Hidden numerical values
            nn.Linear(n_embd, vocab_size)        #Channels -> Vocabulary/Logits
        )
        self.layers = self.emb_linear_stack

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
    bigram_model = EmbeddingProjectionBigramModel(32, 16).to(device)
    
    print("\n\tTraining: ")
    for x in range(1000):
        logits, loss = bigram_model(x_tensor, y_tensor)

        print(f"step {x}: loss {loss}")

    generated_text = "This"

    print("\n\tGenerate: ")

    print(f"Starting text: {generated_text}")
    with torch.no_grad():
        for y in range(10):
            input_token = generated_text[-1]
            input_token_id =  stoi.string_to_ids(vocab, input_token)
            logits = bigram_model(torch.tensor(input_token_id, dtype=torch.long))
            argmax_token_id = torch.argmax(logits).item()
            probabilities = torch.softmax(logits, dim=1)
            softmax_token_id = torch.multinomial(probabilities, num_samples=1).item()

            output_token_id = softmax_token_id

            output_token = itos.ids_to_string(vocab, [output_token_id])

            print(f"step {y}: in_token='{input_token}' (ID:{input_token_id}) | output_token='{output_token}' (ID:{output_token_id})")
            
            generated_text = generated_text+output_token

    print(generated_text)