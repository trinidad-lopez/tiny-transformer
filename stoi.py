import ast

def string_to_ids(vocabulary, string):
    ids = []
    for char in string:
        if char in vocabulary:
            ids.append(vocabulary.index(char) )

    return ids

if __name__ == "__main__":
    print(string_to_ids(ast.literal_eval(input("Vocabulary sorted: ")),input("String to convert: ")))
