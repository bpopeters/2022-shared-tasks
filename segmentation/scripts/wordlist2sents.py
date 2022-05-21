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
        gold_surface, gold_segments = line.rstrip().split("\t")
        out_morphs = []
        for token in gold_surface:
            out_morphs.append(morph_dict.get(token, token))
        sys.stdout.write(" ".join(out_morphs) + "\t" + gold_segments + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
