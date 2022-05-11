import sys

def character_tokenize(string):
    string = string.replace(" ", "_")
    return list(string.strip())


for line in sys.stdin:
    sys.stdout.write(" ".join(character_tokenize(line)) + "\n")
