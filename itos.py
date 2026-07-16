import ast
from collections.abc import Iterable

def ids_to_string(vocabulary, arg):
    string = ""

    if isinstance(arg, Iterable) and not isinstance(arg, (str, bytes, bytearray)):
        iteration = arg
    else:
        iteration = [arg]

    for id in iteration:
        string += vocabulary[id]

    return string

if __name__ == "__main__":
    print(ids_to_string(ast.literal_eval(input("Vocabulary sorted: ")), ast.literal_eval(input("IDs to convert: "))))
