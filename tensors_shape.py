import torch
import ast
import input_target
import stoi
import itos


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
    print(f"x tensor: \n\t{x_tensor} ")
    print(f"y tensor: \n\t{y_tensor} ")
    print(f"x shape: \n\t{x_tensor.shape} ")
    print(f"y shape: \n\t{y_tensor.shape} ")
    print(f"x dtype: \n\t{x_tensor.dtype} ")
    print(f"y dtype: \n\t{y_tensor.dtype} \n\n")

    x2_s = itos.ids_to_string(vocab, x_tensor[2])
    y2_s = itos.ids_to_string(vocab, y_tensor[2])
    print(f"x[2] -> \"{x2_s}\" ")
    print(f"y[2] -> \"{y2_s}\" ")