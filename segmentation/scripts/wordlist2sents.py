import sys
from collections import defaultdict


def main(dict_path):
    morph_dict = defaultdict(set)
    with open(dict_path) as f:
        for line in f:
            try:
                surface, segments = line.strip().split("\t")
            except ValueError:
                sys.stderr.write("bad line: " + line)
            morph_dict[surface].add(segments)
    assert all(len(v) == 1 for v in morph_dict.values())
    morph_dict = {k: next(iter(v)) for k, v in morph_dict.items()}
    for line in sys.stdin:
        out_morphs = []
        for token in line.strip().split():
            out_morphs.append(morph_dict.get(token, token))
        sys.stdout.write(" ".join(out_morphs) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
