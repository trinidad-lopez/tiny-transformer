import ast

def ids_to_string(vocabulary, ids):
    string = ""
    for id in ids:
        string += vocabulary[id]

    return string

if __name__ == "__main__":
    print(ids_to_string(ast.literal_eval(input("Vocabulary sorted: ")), ast.literal_eval(input("IDs to convert: "))))
