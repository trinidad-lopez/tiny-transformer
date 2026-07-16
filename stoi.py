import ast
from collections.abc import Iterable
import copy

def string_to_ids(vocabulary, arg):
    ids = []
    if isinstance(arg, Iterable) and not isinstance(arg, (str, bytes, bytearray)):
        iteration = arg
    else:
        iteration = [arg]
    
    n = 0
    for x in iteration:
        ids.insert(n+1, [])
        for char in x:
            if char in vocabulary:
                ids[n].append(vocabulary.index(char))
        
        n+=1

    return ids

if __name__ == "__main__":
    print(string_to_ids(ast.literal_eval(input("Vocabulary sorted: ")),input("String to convert: ")))
