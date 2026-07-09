def unique_chars_fn(text_file):
    with open(text_file, 'r', encoding= 'utf-8') as f:
        unique_chars = sorted(set(f.read()))

    return unique_chars


if __name__ == "__main__":
    print(unique_chars_fn(input("Path/to/text.txt: ")))