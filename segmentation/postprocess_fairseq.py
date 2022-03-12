#!/usr/bin/env python

import sys
import re

def main():
    for line in sys.stdin:
        no_space = "".join(line.strip().split()) + "\n"
        out_line = re.sub(r'@', ' @@', no_space)
        sys.stdout.write(out_line)


if __name__ == "__main__":
    main()
    
