import ast
import stoi

def input_target_pair(block_size, batch_size, text : str):
    s = text
    x = {}
    y = {}
    for i in range(batch_size):
        if((i+block_size+1)>len(text)):
            print("Text too short for B*T size")
            return x, y
        x[i] = s[i:i+block_size]
        y[i] = s[i+1:i+block_size+1]
    return x, y


if __name__ == "__main__":
    block_size = int(input("Block Size: "))
    batch_size = int(input("Batch Size: "))
    vocab = ast.literal_eval(input("Vocabulary sorted: "))
    text = input("Text: ")
    input_s, target_s = input_target_pair(block_size, batch_size , text)
    print(input_s)
    print(target_s)
    id_input = [stoi.string_to_ids(vocab, input_s[i]) for i in range(batch_size)]
    id_target = [stoi.string_to_ids(vocab, target_s[i]) for i in range(batch_size)]
    print(id_input)
    print(id_target)
    #print("Position\tx char\t\tx ID\t\ty char\t\ty ID\n")
    #for i in range(len(x)):
        #print(str(i)+"\t\t"+input_s[i]+"\t\t"+str(x[i])+"\t\t"+target_s[i]+"\t\t"+str(y[i])+"\n")
